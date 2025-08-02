export CUDA_VISIBLE_DEVICES=7
# export WANDB_MODE=offline # enable if your server cannot connect to wandb

python scr/main_minibatch.py > script/run_train.log 2>&1 &