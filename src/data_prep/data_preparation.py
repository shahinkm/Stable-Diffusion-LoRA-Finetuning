import math
import random

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class ImageTextSDTensorDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, transforms, split="train"):
        super().__init__()
        self._tokenizer = tokenizer
        self._transforms = transforms
        self._split = split

        dataset = hf_dataset
        self._tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        self._tokenized_dataset = self._tokenized_dataset.remove_columns(["texts"])

    def __len__(self):
        return len(self._tokenized_dataset)

    def __getitem__(self, index):
        data = self._tokenized_dataset[index]
        image = Image.open(data["image_path"]).convert("RGB")

        return self._transforms(image), torch.tensor(data["input_ids"])

    def _tokenize_function(self, examples):
        texts = [
            random.choice(texts) if self._split == "train" else texts[0]
            for texts in examples["texts"]
        ]
        tokenized_texts = self._tokenizer(
            texts,
            max_length=self._tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        examples["input_ids"] = tokenized_texts.input_ids

        return examples


class ImageTextSDXLTensorDataset(Dataset):
    def __init__(
        self, hf_dataset, tokenizer1, tokenizer2, transforms, target_size, split="train"
    ):
        super().__init__()
        self._tokenizer1 = tokenizer1
        self._tokenizer2 = tokenizer2
        self._transforms = transforms
        self._target_size = tuple(target_size)
        self._split = split

        dataset = hf_dataset
        self._tokenized_dataset = dataset.map(self._tokenize_function, batched=True)
        self._tokenized_dataset = self._tokenized_dataset.remove_columns(["texts"])

    def __len__(self):
        return len(self._tokenized_dataset)

    def __getitem__(self, index):
        data = self._tokenized_dataset[index]
        image = Image.open(data["image_path"]).convert("RGB")
        transformed_image, crop_coords = self._transforms(image)
        add_time_ids = list(
            (image.height, image.width) + crop_coords + self._target_size
        )

        return (
            transformed_image,
            torch.tensor(data["input_ids1"]),
            torch.tensor(data["input_ids2"]),
            torch.tensor(add_time_ids),
        )

    def _tokenize_function(self, examples):
        texts = [
            random.choice(texts) if self._split == "train" else texts[0]
            for texts in examples["texts"]
        ]
        tokenized_texts1 = self._tokenizer1(
            texts,
            max_length=self._tokenizer1.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_texts2 = self._tokenizer2(
            texts,
            max_length=self._tokenizer2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        examples["input_ids1"] = tokenized_texts1.input_ids
        examples["input_ids2"] = tokenized_texts2.input_ids

        return examples

class ImageTextDataset(Dataset):
    def __init__(self, hf_dataset, transforms):
        super().__init__()
        self._hf_dataset = hf_dataset
        self._transforms = transforms

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, index):
        data = self._hf_dataset[index]
        image = Image.open(data["image_path"]).convert("RGB")

        return (np.array(self._transforms(image)), data["texts"][0])


class DreamboothClassPromptDataset(Dataset):
    def __init__(self, prompts, num_samples):
        super().__init__()
        self._prompts = prompts
        self._num_samples = num_samples

        self._prompts = self._prompts * math.ceil(num_samples / len(prompts))
        self._prompts = self._prompts[:num_samples]

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        return f"{index}-prior-generated.jpg", self._prompts[index]


class ResizeLargestSide:
    def __init__(self, size):
        self._size = size

    def __call__(self, img):
        width, height = img.size
        aspect_ratio = width / height
        if width > height:
            target_width = self._size
            target_height = round(self._size / aspect_ratio)
        else:
            target_height = self._size
            target_width = round(self._size * aspect_ratio)

        resized_img = F.resize(img, (target_height, target_width))
        return resized_img


class PadToSize:
    def __init__(self, size):
        self._size = size

        if isinstance(size, int):
            self._size = (size, size)

    def __call__(self, img):
        pad_width = max(round((self._size[0] - img.width) / 2), 0)
        pad_height = max(round((self._size[1] - img.height) / 2), 0)
        fill = img.getpixel((0, 0))
        return F.pad(img, (pad_width, pad_height), fill)


class RandomCropWithCoords:
    def __init__(self, size):
        if isinstance(size, int):
            self._height = self._width = size
        else:
            try:
                if len(size) == 2:
                    self._height, self._width = size
                else:
                    raise ValueError
            except TypeError:
                raise ValueError(
                    "Size should be an integer or an iterable with two elements"
                )

    def __call__(self, img):
        y, x, h, w = transforms.RandomCrop.get_params(img, (self._height, self._width))
        img = F.crop(img, y, x, h, w)
        return img, (y, x)


class CenterCropWithCoords:
    def __init__(self, size):
        if isinstance(size, int):
            self._height = self._width = size
        else:
            try:
                if len(size) == 2:
                    self._height, self._width = size
                else:
                    raise ValueError
            except TypeError:
                raise ValueError(
                    "Size should be an integer or an iterable with two elements"
                )

        self._center_crop = transforms.CenterCrop((self._height, self._width))

    def __call__(self, img):
        y = max(0, int(round((img.height - self._height) / 2.0)))
        x = max(0, int(round((img.width - self._width) / 2.0)))
        img = self._center_crop(img)
        return img, (y, x)


class ComposeWithCropCoords:
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, img):
        crop_coords = None

        for t in self._transforms:
            if isinstance(t, RandomCropWithCoords) or isinstance(
                t, CenterCropWithCoords
            ):
                img, crop_coords = t(img)
            else:
                img = t(img)

        if crop_coords:
            return img, crop_coords

        return img
