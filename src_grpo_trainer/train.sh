export CUDA_VISIBLE_DEVICES=1,2,3,5  # Set the GPUs to use
export WANDB_MODE=offline  # Disable Weights & Biases logging
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
export RUN_NAME=default-GRPO-fixed_format_reward
export OUTPUT_DIR=outputs/${RUN_NAME}

accelerate launch train_grpo.py \
	--config_file default_config.yaml \