import torch.nn as nn


class LoraWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_config(cls, base_model, unet_lora_config, text_encoder_lora_config=None):
        lora_model = cls(base_model)
        lora_model.base_model.add_lora_adapter(
            unet_lora_config, text_encoder_lora_config
        )
        return lora_model

    @classmethod
    def from_pretrained(cls, base_model, lora_path):
        lora_model = cls(base_model)
        lora_model.load_pretrained(lora_path)
        return lora_model

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def parameters(self):
        return filter(lambda p: p.requires_grad, self.base_model.parameters())

    def save_pretrained(self, path):
        self.base_model.save_pretrained(path)

    def load_pretrained(self, path):
        self.base_model.load_pretrained(path)

    def enable_xformers(self):
        self.base_model.enable_xformers()

    def enable_gradient_checkpointing(self):
        self.base_model.enable_gradient_checkpointing()
