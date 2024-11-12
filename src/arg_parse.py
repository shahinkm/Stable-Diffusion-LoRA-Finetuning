import argparse


def parse_train_args(args=None):
    parser = argparse.ArgumentParser(description="Model Training Inputs")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/train_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the data is stored",
    )
    parser.add_argument(
        "--model_log_dir",
        type=str,
        default="./model_logs",
        help="Directory where the models are/to be saved",
    )
    parser.add_argument(
        "--restore_version",
        type=str,
        default=None,
        help="Version to restore for training continuation",
    )
    return parser.parse_args(args)


def parse_eval_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluation Inputs")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/eval_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the data is stored",
    )
    parser.add_argument(
        "--model_log_dir",
        type=str,
        default="./model_logs",
        help="Directory where the models are/to be saved",
    )
    parser.add_argument(
        "--restore_version",
        type=str,
        default=None,
        help="Version to restore for training continuation",
    )
    parser.add_argument(
        "--evaluate_base_model",
        action="store_true",
        help="Evaluate the base model if this flag is set",
    )
    return parser.parse_args(args)
