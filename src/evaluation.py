import os
import time
import logging

from tqdm import tqdm
import torch
from accelerate import Accelerator
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore

from src import Generator, set_logger, clear_handlers, precision_type


class Evaluator:
    def __init__(self, generator, config, model_log_dir):
        self._generator = generator
        self._config = config
        self._model_log_dir = model_log_dir

        self._accelerator = Accelerator(
            mixed_precision=self._config.ACCELERATOR.MIXED_PRECISION
        )

    @classmethod
    def for_pretrained_lora(cls, config, model_log_dir, restore_version):
        config.add("RESTORED_FROM", restore_version)
        config.add("FOR_BASE_MODEL", False)
        generator = Generator.for_pretrained_lora(
            model_log_dir,
            restore_version,
            requires_safety_checker=False,
            disable_progress_bar=True,
            dtype=precision_type[config.ACCELERATOR.MIXED_PRECISION],
            enable_xformers=config.MODEL.ENABLE_XFORMERS,
            refiner_name=config.GENERATION.REFINER_NAME,
        )

        return cls(generator, config, model_log_dir)

    @classmethod
    def for_base_model(cls, config, model_log_dir):
        generator = Generator.for_base_model(
            config.MODEL.BASE_MODEL_NAME,
            requires_safety_checker=False,
            disable_progress_bar=True,
            dtype=precision_type[config.ACCELERATOR.MIXED_PRECISION],
            enable_xformers=config.MODEL.ENABLE_XFORMERS,
            refiner_name=config.GENERATION.REFINER_NAME,
        )
        config.add("FOR_BASE_MODEL", True)

        return cls(generator, config, model_log_dir)

    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        torch.manual_seed(self._config.GENERATION.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.GENERATION.SEED)

        if self._accelerator.is_main_process:
            log_dir = self._create_log_dirs()
            eval_log_path = os.path.join(log_dir, "eval_logs", "eval.log")
            eval_logger = set_logger(eval_log_path)
            self._config.save(os.path.join(log_dir, "hyper_params/params.json"))

        eval_dataloader = self._accelerator.prepare(eval_dataloader)

        self._generator.to(self._accelerator.device)

        fid = FrechetInceptionDistance()
        fid.to(self._accelerator.device)

        clip = CLIPScore()
        clip.to(self._accelerator.device)

        with tqdm(
            total=len(eval_dataloader),
            desc="Evaluating",
            disable=not self._accelerator.is_local_main_process,
        ) as t:
            for eval_batch in eval_dataloader:
                real_images, prompts = eval_batch

                real_images = real_images.permute(0, 3, 1, 2)
                prompts = list(prompts)
                fake_images = torch.tensor(
                    self._generator(
                        prompts,
                        best_of_n=self._config.GENERATION.BEST_OF_N,
                        seed=self._config.GENERATION.SEED,
                        output_type="np",
                        refine_frac=self._config.GENERATION.REFINE_FRAC,
                        height=self._config.GENERATION.TARGET_RESOLUTION[0],
                        width=self._config.GENERATION.TARGET_RESOLUTION[1],
                        guidance_scale=self._config.GENERATION.GUIDANCE_SCALE,
                        negative_prompt="drawing, cartoon, painting, illustration, 3d, render, cgi",
                    ),
                    device=self._accelerator.device,
                ).permute(0, 3, 1, 2)

                fid.update(real_images, real=True)
                fid.update(fake_images, real=False)

                clip.update(fake_images, prompts)

                t.update()

        self._gather_fid_features(fid)
        fid_score = fid.compute().item()

        self._gather_clip_score_features(clip)
        clip_score = clip.compute().item()

        if self._accelerator.is_main_process:
            logging.info(
                f"- Validation metrics: fid score: {fid_score:.3f}, clip score: {clip_score:.3f}"
            )
            clear_handlers(eval_logger)

    def _create_log_dirs(self):
        log_dir = os.path.join(
            self._model_log_dir,
            time.strftime("eval_%y%m%d%H%M%S", time.localtime(time.time())),
        )
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, "hyper_params"))
        os.mkdir(os.path.join(log_dir, "eval_logs"))

        return log_dir

    def _gather_fid_features(self, fid):
        fid_feature_list = [
            "real_features_sum",
            "real_features_cov_sum",
            "real_features_num_samples",
            "fake_features_sum",
            "fake_features_cov_sum",
            "fake_features_num_samples",
        ]
        self._gather_features(fid, fid_feature_list)

    def _gather_clip_score_features(self, clip):
        clip_feature_list = ["score", "n_samples"]
        self._gather_features(clip, clip_feature_list)

    def _gather_features(self, metric, feature_list):
        for feature in feature_list:
            feature_tensor = getattr(metric, feature).unsqueeze(0)
            gathered_feature_tensor = (
                self._accelerator.gather(feature_tensor).sum(dim=0).squeeze(0)
            )
            setattr(metric, feature, gathered_feature_tensor)
