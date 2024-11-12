import os
from PIL import Image

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from src import (
    create_dreambooth_hf_dataset,
    DreamboothClassPromptDataset,
    AutoStableDiffusionModel,
    LoraWrapper,
    Params,
    precision_type,
    load_json_to_dict,
    save_dict_to_json,
)


class Generator:
    def __init__(self, pipeline, refiner=None):
        self._pipeline = pipeline
        self._refiner = refiner

        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self._clip_model.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    @classmethod
    def for_pretrained_lora(
        cls,
        model_log_dir,
        restore_version,
        best_or_last="best",
        requires_safety_checker=True,
        disable_progress_bar=False,
        dtype=torch.float32,
        enable_xformers=False,
        refiner_name=None,
    ):
        pipeline, refiner = AutoDiffusionPipeline.from_pretrained_lora(
            model_log_dir,
            restore_version,
            best_or_last=best_or_last,
            requires_safety_checker=requires_safety_checker,
            disable_progress_bar=disable_progress_bar,
            dtype=dtype,
            enable_xformers=enable_xformers,
            refiner_name=refiner_name,
        )

        return cls(pipeline, refiner)

    @classmethod
    def for_base_model(
        cls,
        base_model_name,
        requires_safety_checker=True,
        disable_progress_bar=False,
        dtype=torch.float32,
        enable_xformers=False,
        refiner_name=None,
    ):
        pipeline, refiner = AutoDiffusionPipeline.from_pretrained(
            base_model_name,
            requires_safety_checker=requires_safety_checker,
            disable_progress_bar=disable_progress_bar,
            dtype=dtype,
            enable_xformers=enable_xformers,
            refiner_name=refiner_name,
        )

        return cls(pipeline, refiner)

    def __call__(
        self,
        prompts,
        best_of_n=1,
        seed=None,
        output_type="pil",
        refine_frac=None,
        **kwargs,
    ):
        generated_images = self._generate_images(
            prompts, best_of_n, seed, refine_frac=refine_frac, **kwargs
        )

        if best_of_n > 1:
            _, h, w, c = generated_images.shape
            generated_images = generated_images.reshape(-1, best_of_n, h, w, c)
            generated_images = self._select_best_clip_images(prompts, generated_images)

        if output_type == "pil":
            generated_images = [
                Image.fromarray(image_array) for image_array in generated_images
            ]

        return generated_images

    def to(self, device):
        self._pipeline = self._pipeline.to(device)
        self._clip_model = self._clip_model.to(device)
        if self._refiner:
            self._refiner = self._refiner.to(device)

    def _generate_images(
        self,
        prompts,
        best_of_n,
        seed,
        refine_frac=None,
        num_inference_steps=50,
        **kwargs,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        generator = (
            torch.Generator(device=self._pipeline.device).manual_seed(seed)
            if seed is not None
            else None
        )
        if self._refiner:
            if refine_frac is None:
                refine_frac = 0.0
            if isinstance(self._pipeline, StableDiffusionXLPipeline):
                generated_interim = self._pipeline(
                    prompts,
                    denoising_end=1.0 - refine_frac,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=best_of_n,
                    generator=generator,
                    output_type="latent",
                    **kwargs,
                ).images
                generated_images = self._refiner(
                    prompts,
                    denoising_start=1.0 - refine_frac,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=best_of_n,
                    generator=generator,
                    output_type="np",
                    image=generated_interim,
                    **kwargs,
                ).images
            elif isinstance(self._pipeline, StableDiffusionPipeline):
                generated_interim = self._pipeline(
                    prompts,
                    num_images_per_prompt=best_of_n,
                    num_inference_steps=int((1.0 - refine_frac) * num_inference_steps),
                    generator=generator,
                    output_type="pil",
                    **kwargs,
                ).images
                generated_images = self._refiner(
                    prompts,
                    num_images_per_prompt=best_of_n,
                    num_inference_steps=num_inference_steps,
                    strength=refine_frac,
                    generator=generator,
                    output_type="np",
                    image=generated_interim,
                    **kwargs,
                ).images
            else:
                raise ValueError("Unsupported pipeline type.")
        else:
            generated_images = self._pipeline(
                prompts,
                num_images_per_prompt=best_of_n,
                generator=generator,
                output_type="np",
                **kwargs,
            ).images
        generated_images = (generated_images * 255).astype(np.uint8)

        return generated_images

    @torch.no_grad()
    def _select_best_clip_images(self, prompts, images_grid):
        n_h, _, h, w, c = images_grid.shape
        selected_images = np.zeros((n_h, h, w, c), dtype=np.uint8)
        for i, (prompt, image_row) in enumerate(zip(prompts, images_grid)):
            inputs = self._clip_processor(
                text=prompt, images=image_row, return_tensors="pt", padding=True
            )
            inputs = {
                key: value.to(self._clip_model.device) for key, value in inputs.items()
            }
            logits = self._clip_model(**inputs).logits_per_text[0]
            best_index = logits.argmax().item()
            selected_images[i] = image_row[best_index]

        return selected_images


class AutoDiffusionPipeline:
    @staticmethod
    def from_pretrained_lora(
        model_log_dir,
        restore_version,
        best_or_last="best",
        requires_safety_checker=True,
        disable_progress_bar=False,
        dtype=torch.float32,
        enable_xformers=False,
        refiner_name=None,
    ):
        model_path = os.path.join(model_log_dir, restore_version)
        config = Params(os.path.join(model_path, "hyper_params/params.json"))
        base_model = AutoStableDiffusionModel.from_pretrained(
            config.MODEL.BASE_MODEL_NAME
        )
        lora_model = LoraWrapper.from_pretrained(
            base_model, os.path.join(model_path, f"state/model_{best_or_last}")
        )
        lora_model.eval()

        pipeline, refiner = AutoDiffusionPipeline._get_base_pipelines(
            config.MODEL.BASE_MODEL_NAME,
            requires_safety_checker,
            disable_progress_bar,
            refiner_name,
        )
        if lora_model.base_model.train_unet:
            pipeline.unet = lora_model.base_model.lora_unet
        if lora_model.base_model.train_text_encoder:
            if isinstance(pipeline, StableDiffusionXLPipeline):
                pipeline.text_encoder = lora_model.base_model.lora_text_encoder1
                pipeline.text_encoder_2 = lora_model.base_model.lora_text_encoder2
            elif isinstance(pipeline, StableDiffusionPipeline):
                pipeline.text_encoder = lora_model.base_model.lora_text_encoder
        pipeline = pipeline.to(dtype)
        if refiner:
            refiner = refiner.to(dtype)
        if enable_xformers:
            pipeline.enable_xformers_memory_efficient_attention()
            if refiner:
                refiner.enable_xformers_memory_efficient_attention()

        return pipeline, refiner

    @staticmethod
    def from_pretrained(
        model_name,
        requires_safety_checker=True,
        disable_progress_bar=False,
        dtype=torch.float32,
        enable_xformers=False,
        refiner_name=None,
    ):
        pipeline, refiner = AutoDiffusionPipeline._get_base_pipelines(
            model_name,
            requires_safety_checker,
            disable_progress_bar,
            refiner_name,
        )
        pipeline = pipeline.to(dtype)
        if refiner:
            refiner = refiner.to(dtype)
        if enable_xformers:
            pipeline.enable_xformers_memory_efficient_attention()
            if refiner:
                refiner.enable_xformers_memory_efficient_attention()
        return pipeline, refiner

    def _get_base_pipelines(
        model_name, requires_safety_checker, disable_progress_bar, refiner_name
    ):
        refiner = None

        diffuser_config = DiffusionPipeline.load_config(model_name)
        if diffuser_config["_class_name"] == "StableDiffusionPipeline":
            pipeline = DiffusionPipeline.from_pretrained(
                model_name, requires_safety_checker=requires_safety_checker
            )
            if not requires_safety_checker:
                pipeline.safety_checker = None
            if refiner_name:
                refiner = DiffusionPipeline.from_pretrained(refiner_name)
        elif diffuser_config["_class_name"] == "StableDiffusionXLPipeline":
            pipeline = DiffusionPipeline.from_pretrained(model_name)
            if refiner_name:
                refiner = DiffusionPipeline.from_pretrained(
                    refiner_name,
                    text_encoder_2=pipeline.text_encoder_2,
                    vae=pipeline.vae,
                )
        else:
            raise ValueError("Unsupported pipeline type.")

        if disable_progress_bar:
            pipeline.set_progress_bar_config(disable=disable_progress_bar)
            if refiner_name:
                refiner.set_progress_bar_config(disable=disable_progress_bar)

        return pipeline, refiner


class DreamboothClassDataGenerator:
    def __init__(self, data_dir, config, model_log_dir=None, restore_version=None):
        self._data_dir = data_dir
        self._config = config
        self._model_log_dir = model_log_dir
        self._restore_version = restore_version

        hf_dataset = create_dreambooth_hf_dataset(data_dir, instances_only=True)
        self._prompts = [
            texts[0].replace("<spl>", "").replace("  ", " ")
            for texts in hf_dataset["texts"]
        ]
        self._prompt_dataset = DreamboothClassPromptDataset(
            self._prompts, self._config.DREAMBOOTH.PRIOR_PRESERVATION.NUM_CLASS_IMAGES
        )
        self._captions = load_json_to_dict(
            os.path.join(self._data_dir, "captions/captions.json")
        )
        self._captions = {k: v for k, v in self._captions.items() if "<spl>" in v}

    def generate(self):
        self._prepare_for_generation()
        for filenames, prompts in tqdm(
            self._prompt_dataloader, desc="Generating class images"
        ):
            images = self._generator(
                list(prompts),
                height=self._config.DATA_AUGMENTATION.TARGET_RESOLUTION[0],
                width=self._config.DATA_AUGMENTATION.TARGET_RESOLUTION[1],
                refine_frac=0.2,
            )

            for i, image in enumerate(images):
                image_filename = os.path.join(self._data_dir, "images", filenames[i])
                image.save(image_filename)

        prior_pres_captions = {}
        for i in range(len(self._prompt_dataset)):
            filename, caption = self._prompt_dataset[i]
            prior_pres_captions[filename] = caption

        save_dict_to_json(
            self._captions | prior_pres_captions,
            os.path.join(self._data_dir, "captions/captions.json"),
        )

        self._release_resources()

    def _prepare_for_generation(self):
        if self._restore_version is None:
            self._generator = Generator.for_base_model(
                self._config.MODEL.BASE_MODEL_NAME,
                requires_safety_checker=False,
                disable_progress_bar=True,
                dtype=precision_type[
                    self._config.DREAMBOOTH.PRIOR_PRESERVATION.MIXED_PRECISION
                ],
                enable_xformers=self._config.DREAMBOOTH.PRIOR_PRESERVATION.ENABLE_XFORMERS,
                refiner_name="stabilityai/stable-diffusion-xl-refiner-1.0",
            )
        else:
            self._generator = Generator.for_pretrained_lora(
                self._model_log_dir,
                self._restore_version,
                requires_safety_checker=False,
                disable_progress_bar=True,
                dtype=precision_type[
                    self._config.DREAMBOOTH.PRIOR_PRESERVATION.MIXED_PRECISION
                ],
                enable_xformers=self._config.DREAMBOOTH.PRIOR_PRESERVATION.ENABLE_XFORMERS,
                refiner_name="stabilityai/stable-diffusion-xl-refiner-1.0",
            )
        self._prompt_dataloader = DataLoader(
            self._prompt_dataset,
            batch_size=self._config.DREAMBOOTH.PRIOR_PRESERVATION.BATCH_SIZE,
        )
        if torch.cuda.is_available:
            self._generator.to("cuda:1")

    def _release_resources(self):
        del self._generator
        del self._prompt_dataloader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
