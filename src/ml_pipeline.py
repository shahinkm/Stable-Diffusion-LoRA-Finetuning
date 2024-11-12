import os

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig

from src import (
    create_hf_dataset,
    create_dreambooth_hf_dataset,
    DreamboothClassDataGenerator,
    ImageTextSDTensorDataset,
    ImageTextSDXLTensorDataset,
    ImageTextDataset,
    RandomCropWithCoords,
    CenterCropWithCoords,
    ComposeWithCropCoords,
    AutoStableDiffusionModel,
    LoraWrapper,
    Trainer,
    Evaluator,
    Params,
    convert_to_lora_target_names,
)


class StableDiffusionMLPipeline:
    def __init__(self, args, mode):
        self._mode = mode

        self._config = Params(args.config_path)

        if self._mode == "train":
            self._train_dataloader, self._val_dataloader = self._prepare_train_data(
                args
            )
            self._trainer = self._get_trainer(args)
        elif self._mode == "eval":
            self._eval_dataloader = self._prepare_eval_data(args)
            if args.evaluate_base_model:
                self._evaluator = Evaluator.for_base_model(
                    self._config, args.model_log_dir
                )
            else:
                self._evaluator = Evaluator.for_pretrained_lora(
                    self._config, args.model_log_dir, args.restore_version
                )

    @classmethod
    def for_training(cls, args):
        return cls(args, "train")

    @classmethod
    def for_evaluation(cls, args):
        return cls(args, "eval")

    def run(self):
        if self._mode == "train":
            self._trainer.train(self._train_dataloader, self._val_dataloader)
        elif self._mode == "eval":
            self._evaluator.evaluate(self._eval_dataloader)

    def _get_trainer(self, args):
        base_model = AutoStableDiffusionModel.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME
        )
        unet_lora_config = (
            LoraConfig(
                r=self._config.MODEL.UNET_LORA.RANK,
                lora_alpha=self._config.MODEL.UNET_LORA.ALPHA,
                target_modules=convert_to_lora_target_names(
                    self._config.MODEL.UNET_LORA.TARGET, "unet"
                ),
                init_lora_weights="gaussian",
            )
            if self._config.MODEL.TRAIN_UNET
            else None
        )
        text_encoder_lora_config = (
            LoraConfig(
                r=self._config.MODEL.TEXT_ENCODER_LORA.RANK,
                lora_alpha=self._config.MODEL.TEXT_ENCODER_LORA.ALPHA,
                target_modules=convert_to_lora_target_names(
                    self._config.MODEL.TEXT_ENCODER_LORA.TARGET, "text_encoder"
                ),
                init_lora_weights="gaussian",
            )
            if self._config.MODEL.TRAIN_TEXT_ENCODER
            else None
        )
        lora_model = LoraWrapper.from_config(
            base_model, unet_lora_config, text_encoder_lora_config
        )
        if args.restore_version is not None:
            model_path = os.path.join(
                args.model_log_dir, args.restore_version, "state/model_best"
            )
            lora_model.load_pretrained(model_path)
            self._config.add("RESTORED_FROM", args.restore_version)
        if self._config.MODEL.ENABLE_XFORMERS:
            lora_model.enable_xformers()
        if self._config.MODEL.ENABLE_GRADIENT_CHECKPOINTING:
            lora_model.enable_gradient_checkpointing()

        optimizer = AdamW(
            lora_model.parameters(),
            lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
            betas=(
                self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
            ),
            weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
            eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
        )
        lr_scheduler = get_scheduler(
            self._config.TRAINING.LR_SCHEDULER.TYPE,
            optimizer=optimizer,
            num_warmup_steps=self._config.TRAINING.LR_SCHEDULER.WARMUP_STEPS
            // self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS,
            num_training_steps=self._config.TRAINING.EPOCHS
            * len(self._train_dataloader)
            // self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS,
        )

        trainer = Trainer(
            lora_model,
            optimizer,
            lr_scheduler,
            self._config,
            args.model_log_dir,
        )

        return trainer

    def _prepare_train_data(self, args):
        hf_dataset = create_hf_dataset(
            args.data_dir,
            self._config.TRAINING.TEST_SIZE,
            min_aspect_ratio=self._config.DATA_AUGMENTATION.MIN_ASPECT_RATIO,
            max_aspect_ratio=self._config.DATA_AUGMENTATION.MAX_ASPECT_RATIO,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer"
        )

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION),
                (
                    transforms.RandomCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA_AUGMENTATION.RANDOM_CROP
                    else transforms.CenterCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataset = ImageTextSDTensorDataset(
            hf_dataset["train"], tokenizer, train_transforms, split="train"
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    min(self._config.DATA_AUGMENTATION.TARGET_RESOLUTION)
                ),
                transforms.CenterCrop(self._config.DATA_AUGMENTATION.TARGET_RESOLUTION),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        val_dataset = ImageTextSDTensorDataset(
            hf_dataset["validation"], tokenizer, val_transforms, split="validation"
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._config.TRAINING.BATCH_SIZE.TEST, shuffle=False
        )

        return train_dataloader, val_dataloader

    def _prepare_eval_data(self, args):
        eval_hf_dataset = create_hf_dataset(
            args.data_dir,
            min_aspect_ratio=self._config.DATA.MIN_ASPECT_RATIO,
            max_aspect_ratio=self._config.DATA.MAX_ASPECT_RATIO,
        )
        if self._config.DATA.N_SAMPLES:
            eval_hf_dataset = eval_hf_dataset.select(range(self._config.DATA.N_SAMPLES))

        eval_transforms = transforms.Compose(
            [
                transforms.Resize(min(self._config.GENERATION.TARGET_RESOLUTION)),
                transforms.CenterCrop(self._config.GENERATION.TARGET_RESOLUTION),
            ]
        )
        eval_dataset = ImageTextDataset(eval_hf_dataset, eval_transforms)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self._config.EVALUATION.BATCH_SIZE, shuffle=False
        )

        return eval_dataloader


class StableDiffusionXLMLPipeline(StableDiffusionMLPipeline):
    def __init__(self, args, mode):
        super().__init__(args, mode)

    def _prepare_train_data(self, args):
        hf_dataset = create_hf_dataset(
            args.data_dir,
            self._config.TRAINING.TEST_SIZE,
            min_aspect_ratio=self._config.DATA_AUGMENTATION.MIN_ASPECT_RATIO,
            max_aspect_ratio=self._config.DATA_AUGMENTATION.MAX_ASPECT_RATIO,
        )
        tokenizer1 = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer"
        )
        tokenizer2 = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer_2"
        )

        train_transforms = ComposeWithCropCoords(
            [
                transforms.Resize(self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION),
                (
                    RandomCropWithCoords(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA_AUGMENTATION.RANDOM_CROP
                    else CenterCropWithCoords(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataset = ImageTextSDXLTensorDataset(
            hf_dataset["train"],
            tokenizer1,
            tokenizer2,
            train_transforms,
            self._config.DATA_AUGMENTATION.TARGET_RESOLUTION,
            split="train",
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        val_transforms = ComposeWithCropCoords(
            [
                transforms.Resize(
                    min(self._config.DATA_AUGMENTATION.TARGET_RESOLUTION)
                ),
                CenterCropWithCoords(self._config.DATA_AUGMENTATION.TARGET_RESOLUTION),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        val_dataset = ImageTextSDXLTensorDataset(
            hf_dataset["validation"],
            tokenizer1,
            tokenizer2,
            val_transforms,
            self._config.DATA_AUGMENTATION.TARGET_RESOLUTION,
            split="validation",
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TEST,
            shuffle=False,
        )

        return train_dataloader, val_dataloader


class StableDiffusionDBMLPipeline(StableDiffusionMLPipeline):
    def __init__(self, args, mode):
        super().__init__(args, mode)

    @classmethod
    def for_evaluation(cls, _):
        raise NotImplementedError(
            "This method is is not applicable for StableDiffusionDBMLPipeline."
        )

    def _prepare_train_data(self, args):
        if self._config.DREAMBOOTH.PRIOR_PRESERVATION is not None:
            dreambooth_prior_generator = DreamboothClassDataGenerator(
                args.data_dir, self._config, args.model_log_dir, args.restore_version
            )
            dreambooth_prior_generator.generate()

        hf_dataset = create_dreambooth_hf_dataset(
            args.data_dir,
            instances_only=(not self._config.DREAMBOOTH.PRIOR_PRESERVATION),
            identifier=self._config.DREAMBOOTH.IDENTIFIER,
            min_aspect_ratio=self._config.DATA_AUGMENTATION.MIN_ASPECT_RATIO,
            max_aspect_ratio=self._config.DATA_AUGMENTATION.MAX_ASPECT_RATIO,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer"
        )

        train_transforms = transforms.Compose(
            [
                transforms.Resize(self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION),
                (
                    transforms.RandomCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA_AUGMENTATION.RANDOM_CROP
                    else transforms.CenterCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataset = ImageTextSDTensorDataset(
            hf_dataset, tokenizer, train_transforms
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        return train_dataloader, None


class StableDiffusionXLDBMLPipeline(StableDiffusionMLPipeline):
    def __init__(self, args, mode):
        super().__init__(args, mode)

    @classmethod
    def for_evaluation(cls, _):
        raise NotImplementedError(
            "This method is is not applicable for StableDiffusionXLDBMLPipeline."
        )

    def _prepare_train_data(self, args):
        prior_preserve = hasattr(self._config.DREAMBOOTH, "PRIOR_PRESERVATION")
        if prior_preserve:
            dreambooth_prior_generator = DreamboothClassDataGenerator(
                args.data_dir, self._config, args.model_log_dir, args.restore_version
            )
            dreambooth_prior_generator.generate()

        hf_dataset = create_dreambooth_hf_dataset(
            args.data_dir,
            instances_only=not prior_preserve,
            identifier=self._config.DREAMBOOTH.IDENTIFIER,
            min_aspect_ratio=self._config.DATA_AUGMENTATION.MIN_ASPECT_RATIO,
            max_aspect_ratio=self._config.DATA_AUGMENTATION.MAX_ASPECT_RATIO,
        )
        tokenizer1 = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer"
        )
        tokenizer2 = AutoTokenizer.from_pretrained(
            self._config.MODEL.BASE_MODEL_NAME, subfolder="tokenizer_2"
        )

        train_transforms = ComposeWithCropCoords(
            [
                transforms.Resize(self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION),
                (
                    RandomCropWithCoords(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA_AUGMENTATION.RANDOM_CROP
                    else CenterCropWithCoords(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataset = ImageTextSDXLTensorDataset(
            hf_dataset,
            tokenizer1,
            tokenizer2,
            train_transforms,
            self._config.DATA_AUGMENTATION.TARGET_RESOLUTION,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        return train_dataloader, None


class AutoMLPipeline:
    @staticmethod
    def for_training(args):
        return AutoMLPipeline._get_pipeline_class(args, "train")

    @staticmethod
    def for_evaluation(args):
        return AutoMLPipeline._get_pipeline_class(args, "eval")

    @staticmethod
    def _get_pipeline_class(args, mode):
        config = Params(args.config_path)
        diffuser_config = DiffusionPipeline.load_config(config.MODEL.BASE_MODEL_NAME)

        if hasattr(config, "DREAMBOOTH"):
            ml_pipeline_map = {
                "StableDiffusionPipeline": StableDiffusionDBMLPipeline,
                "StableDiffusionXLPipeline": StableDiffusionXLDBMLPipeline,
            }
        else:
            ml_pipeline_map = {
                "StableDiffusionPipeline": StableDiffusionMLPipeline,
                "StableDiffusionXLPipeline": StableDiffusionXLMLPipeline,
            }

        ml_pipeline_class = ml_pipeline_map.get(diffuser_config["_class_name"])

        if not ml_pipeline_class:
            raise ValueError(
                f"Unsupported base_model name: {diffuser_config['_class_name']}"
            )

        if mode == "train":
            return ml_pipeline_class.for_training(args)
        elif mode == "eval":
            return ml_pipeline_class.for_evaluation(args)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
