import json
import logging

import matplotlib.pyplot as plt
import torch
import evaluate


class DotDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __contains__(self, item):
        return item in self.keys()


class ConfigBase:
    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
        for key, value in params.items():
            setattr(self, key, self._to_dotdict(value))


class Params(ConfigBase):
    def __init__(self, json_path=None):
        if json_path:
            self.load(json_path)

    def _to_dotdict(self, obj):
        if isinstance(obj, dict):
            return DotDict({k: self._to_dotdict(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self._to_dotdict(item) for item in obj]
        else:
            return obj

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
        self.__dict__ = self._to_dotdict(params)

    def add(self, key, value):
        if "ADDED" not in self.__dict__:
            self.ADDED = DotDict()
        self.ADDED[key] = value


class RunningAverageDict:
    def __init__(self, keys):
        self.total_dict = {}
        for key in keys:
            self.total_dict[key] = 0
        self.steps = 0

    def update(self, val_dict):
        for key in self.total_dict:
            if key.endswith(":c"):
                self.total_dict[key] = val_dict[key]
            else:
                self.total_dict[key] += val_dict[key]
        self.steps += 1

    def serialize(self):
        keys = list(self.total_dict.keys())
        values = torch.tensor([list(self.total_dict.values())], dtype=torch.float32)
        steps = torch.tensor([self.steps], dtype=torch.float32)

        return keys, values, steps

    def reset(self):
        for key in self.total_dict:
            self.total_dict[key] = 0
        self.steps = 0

    def __call__(self):
        return {
            key.split(":")[0] if key.endswith(":c") else key: (
                value if key.endswith(":c") else value / float(self.steps)
            )
            for key, value in self.total_dict.items()
        }


class RougeScore:
    def __init__(self, tokenizer, rouge_type):
        self._tokenizer = tokenizer
        self._rouge_type = rouge_type

        self._rouge = evaluate.load("rouge")

    def __call__(self, predictions, labels):
        decoded_predictions = self._tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        labels[labels == -100] = self._tokenizer.pad_token_id
        decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_score_dict = self._rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_aggregator=True,
            use_stemmer=True,
            rouge_types=[self._rouge_type],
        )
        return rouge_score_dict[self._rouge_type]


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def clear_handlers(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        json.dump(d, f, indent=4)


def load_json_to_dict(json_path):
    with open(json_path, "r") as f:
        dict = json.load(f)

    return dict


def denormalize_batch(tensor_batch, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    tensor_batch = tensor_batch * std + mean
    return tensor_batch


def show_tensor_images(tensors, num=4):
    tensors = tensors.detach().cpu()
    tensors = tensors.permute(0, 2, 3, 1)
    tensors = tensors.clamp(0, 1)

    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(tensors[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def convert_to_lora_target_names(modules, model_type):
    if model_type == "unet":
        target_map = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
    elif model_type == "text_encoder":
        target_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}
    else:
        raise ValueError("Unsupported model type provided.")

    return [target_map[module] for module in modules if module in target_map]


precision_type = {"fp16": torch.float16, "bf16": torch.bfloat16}


def get_aspect_ratio(image):
    width, height = image.size
    return width / height
