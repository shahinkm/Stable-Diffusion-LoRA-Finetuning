import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
    DiffusionPipeline,
)
from peft import PeftModel


class StableDiffusionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self._model_name = model_name

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        self.train_unet = False
        self.train_text_encoder = False
        self.lora_unet = self.unet
        self.lora_text_encoder = self.text_encoder

    def forward(self, images, input_ids, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        encoded_text = self.lora_text_encoder(input_ids).last_hidden_state

        image_latents = self.vae.encode(images).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=image_latents.device,
        )
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)

        noise_pred = self.lora_unet(
            sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoded_text
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    def add_lora_adapter(self, unet_lora_config=None, text_encoder_lora_config=None):
        if unet_lora_config is None and text_encoder_lora_config is None:
            raise ValueError(
                "At least one of `unet_lora_config` or `text_encoder_lora_config` must be provided."
            )

        if unet_lora_config:
            self.train_unet = True
            self.lora_unet = PeftModel(self.unet, unet_lora_config)
        if text_encoder_lora_config:
            self.train_text_encoder = True
            self.lora_text_encoder = PeftModel(
                self.text_encoder, text_encoder_lora_config
            )

    def save_pretrained(self, path):
        if self.train_unet:
            self.lora_unet.save_pretrained(os.path.join(path, "lora_unet"))
        if self.train_text_encoder:
            self.lora_text_encoder.save_pretrained(
                os.path.join(path, "lora_text_encoder")
            )

    def load_pretrained(self, path):
        if "lora_unet" in os.listdir(path):
            self.train_unet = True
            self.lora_unet = PeftModel.from_pretrained(
                self.unet, os.path.join(path, "lora_unet"), is_trainable=True
            )
        if "lora_text_encoder" in os.listdir(path):
            self.train_text_encoder = True
            self.lora_text_encoder = PeftModel.from_pretrained(
                self.text_encoder,
                os.path.join(path, "lora_text_encoder"),
                is_trainable=True,
            )

    def train(self, mode=True):
        if self.train_unet:
            self.lora_unet.train(mode)
        if self.train_text_encoder:
            self.lora_text_encoder.train(mode)

    def enable_xformers(self):
        self.vae.enable_xformers_memory_efficient_attention()
        self.lora_unet.enable_xformers_memory_efficient_attention()

    def enable_gradient_checkpointing(self):
        if self.train_unet:
            self.lora_unet.enable_gradient_checkpointing()


class StableDiffusionXLModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self._model_name = model_name

        self.text_encoder1 = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        )
        self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder_2"
        )
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        self.train_unet = False
        self.train_text_encoder = False
        self.lora_unet = self.unet
        self.lora_text_encoder1 = self.text_encoder1
        self.lora_text_encoder2 = self.text_encoder2

    def forward(self, images, input_ids1, input_ids2, add_time_ids, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        encoded_text1 = self.lora_text_encoder1(input_ids1, output_hidden_states=True)
        encoded_text2 = self.lora_text_encoder2(input_ids2, output_hidden_states=True)

        embedded_text1 = encoded_text1.hidden_states[-2]
        embedded_text2 = encoded_text2.hidden_states[-2]

        pooled_embed2 = encoded_text2.text_embeds
        embedded_text_cat = torch.cat([embedded_text1, embedded_text2], dim=-1)

        image_latents = self.vae.encode(images).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        noise = torch.randn_like(image_latents)
        batch_size = image_latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=image_latents.device,
        )
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(image_latents, noise, timesteps)

        add_time_ids = add_time_ids.to(pooled_embed2.dtype)
        unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_embed2}
        noise_pred = self.lora_unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=embedded_text_cat,
            added_cond_kwargs=unet_added_conditions,
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    def add_lora_adapter(self, unet_lora_config=None, text_encoder_lora_config=None):
        if unet_lora_config is None and text_encoder_lora_config is None:
            raise ValueError(
                "At least one of `unet_lora_config` or `text_encoder_lora_config` must be provided."
            )

        if unet_lora_config:
            self.train_unet = True
            self.lora_unet = PeftModel(self.unet, unet_lora_config)
        if text_encoder_lora_config:
            self.train_text_encoder = True
            self.lora_text_encoder1 = PeftModel(
                self.text_encoder1, text_encoder_lora_config
            )
            self.lora_text_encoder2 = PeftModel(
                self.text_encoder2, text_encoder_lora_config
            )

    def save_pretrained(self, path):
        if self.train_unet:
            self.lora_unet.save_pretrained(os.path.join(path, "lora_unet"))
        if self.train_text_encoder:
            self.lora_text_encoder1.save_pretrained(
                os.path.join(path, "lora_text_encoder1")
            )
            self.lora_text_encoder2.save_pretrained(
                os.path.join(path, "lora_text_encoder2")
            )

    def load_pretrained(self, path):
        if "lora_unet" in os.listdir(path):
            self.train_unet = True
            self.lora_unet = PeftModel.from_pretrained(
                self.unet, os.path.join(path, "lora_unet"), is_trainable=True
            )
        if ("lora_text_encoder1" in os.listdir(path)) and (
            "lora_text_encoder2" in os.listdir(path)
        ):
            self.train_text_encoder = True
            self.lora_text_encoder1 = PeftModel.from_pretrained(
                self.text_encoder1,
                os.path.join(path, "lora_text_encoder1"),
                is_trainable=True,
            )
            self.lora_text_encoder2 = PeftModel.from_pretrained(
                self.text_encoder2,
                os.path.join(path, "lora_text_encoder2"),
                is_trainable=True,
            )

    def train(self, mode=True):
        if self.train_unet:
            self.lora_unet.train(mode)
        if self.train_text_encoder:
            self.lora_text_encoder1.train(mode)
            self.lora_text_encoder2.train(mode)

    def enable_xformers(self):
        self.vae.enable_xformers_memory_efficient_attention()
        self.lora_unet.enable_xformers_memory_efficient_attention()

    def enable_gradient_checkpointing(self):
        if self.train_unet:
            self.lora_unet.enable_gradient_checkpointing()


class AutoStableDiffusionModel:
    @staticmethod
    def from_pretrained(model_name):
        diffuser_config = DiffusionPipeline.load_config(model_name)

        if diffuser_config["_class_name"] == "StableDiffusionPipeline":
            return StableDiffusionModel(model_name)
        elif diffuser_config["_class_name"] == "StableDiffusionXLPipeline":
            return StableDiffusionXLModel(model_name)
        else:
            raise ValueError("Unsupported base_model name")
