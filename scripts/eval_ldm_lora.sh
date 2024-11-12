accelerate launch evaluate_ldm_lora.py \
--config_path ./configs/experiment4/eval_config_sdxl_w_refiner.json \
--data_dir ./data/deep_fashion \
--model_log_dir ./model_logs \
--evaluate_base_model