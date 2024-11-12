import os

import pandas as pd
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from PIL import Image

from src import load_json_to_dict, get_aspect_ratio


def create_hf_dataset(
    data_dir, test_size=None, min_aspect_ratio=None, max_aspect_ratio=None
):
    images_dir = os.path.join(data_dir, "images")

    captions = load_json_to_dict(os.path.join(data_dir, "captions/captions.json"))

    data = []

    for filename in os.listdir(images_dir):
        image_path = os.path.join(images_dir, filename)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            aspect_ratio = get_aspect_ratio(image)

            if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
                continue
            if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
                continue

            caption = captions.get(filename, "No caption")
            if caption != "No caption":
                data.append(
                    {
                        "image_path": image_path,
                        "texts": caption if isinstance(caption, list) else [caption],
                    }
                )

    data_df = pd.DataFrame(data)

    hf_dataset = HFDataset.from_pandas(data_df)

    if test_size:
        hf_dataset = hf_dataset.train_test_split(test_size=test_size)
        hf_dataset["validation"] = hf_dataset.pop("test")

    return hf_dataset


def create_dreambooth_hf_dataset(
    data_dir,
    instances_only=False,
    identifier=None,
    min_aspect_ratio=None,
    max_aspect_ratio=None,
):
    images_dir = os.path.join(data_dir, "images")

    captions = load_json_to_dict(os.path.join(data_dir, "captions/captions.json"))

    if instances_only:
        captions = {k: v for k, v in captions.items() if "<spl>" in v}

    data = []

    for filename in captions:
        caption = captions.get(filename, "No caption")
        if identifier is not None:
            caption = caption.replace("<spl>", identifier)
        image_path = os.path.join(images_dir, filename)

        if os.path.exists(image_path) and caption != "No caption":
            image = Image.open(image_path)
            aspect_ratio = get_aspect_ratio(image)

            if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
                continue
            if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
                continue
            data.append({"image_path": image_path, "texts": [caption]})

    data_df = pd.DataFrame(data)

    hf_dataset = HFDataset.from_pandas(data_df)

    return hf_dataset