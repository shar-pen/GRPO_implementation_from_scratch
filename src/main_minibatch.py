import json, re, torch, copy, wandb
from tqdm import tqdm
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup

from prepare_data import prepare_dataloader
from reward import format_reward, tag_count_reward, accuracy_reward, compute_grpo_reward, compute_group_advantage
from grpo_utils import generate_rollouts, get_per_token_log_probs, get_grpo_loss
from model_utils import optimize_model_settings, freeze_model


n_epoch = 1
n_roullout = 32
max_length = 1024
batch_size_dataloader = 4 # num of batch per step
batch_size_micro = 2 # num of data for backward
batch_size_micro_for_no_grad = 4 # num of data for no grad, to avoid OOM
learning_rate = 5e-6
epsilon = 0.2 # PPO ratio clipping threshold
beta = 0.04 # KL divergence coefficient
mu = 2 # num of policy updates per batch


model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

train_dataloader, eval_dataloader = prepare_dataloader(tokenizer, batch_size=batch_size_dataloader)
reward_funcs = [format_reward, tag_count_reward, accuracy_reward]
reward_weights = [0.5, 0.5, 1.0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_policy = AutoModelForCausalLM.from_pretrained(
	model_name_or_path,
	torch_dtype='auto',
).to(device)
optimize_model_settings(model_policy)


wandb.init(project="GRPO_scratch")
optimizer = torch.optim.AdamW(model_policy.parameters(), lr=learning_rate)

# 计算总的训练步数
total_steps = len(train_dataloader) * n_epoch
warmup_steps = int(total_steps * 0.1)  # 10% warm up
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

step = 0
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

for epoch in range(n_epoch):

	print(f"Epoch {epoch + 1}/{n_epoch}")

	model_reference = copy.deepcopy(model_policy)
	freeze_model(model_reference)

	for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader), leave=True):
	# for batch in train_dataloader:

		prompts = [example['prompt'] for example in batch]
		solutions = [example['solution'] for example in batch]

		sequence_ids, sequence_mask, completion_mask, completions = generate_rollouts(
			model_policy, 
			tokenizer, 
			prompts, 
			num_of_roullout=n_roullout, 
			max_length=max_length, 
			temperature=1.0, 
			top_p=0.9, 
			top_k=50,
		)

		
		# 将 solutions 扩展到与 completions 相同
		solutions = [s for s in solutions for _ in range(n_roullout)]

		reward_per_completion, reward_per_reward_func = compute_grpo_reward(
			completions, 
			solutions, 
			reward_funcs,
			reward_weights,
		)

		group_advantage_per_sample = compute_group_advantage(
			reward_per_completion
		).to(device)

		# 在训练前一次性计算所有的old_policy_prob和reference_prob (使用mini batch避免显存问题)
		with torch.no_grad():
			prob_per_token_old = []
			prob_per_token_reference = []
			
			for i in range(0, len(sequence_ids), batch_size_micro_for_no_grad):
				sequence_ids_batch = sequence_ids[i:i + batch_size_micro_for_no_grad]
				sequence_mask_batch = sequence_mask[i:i + batch_size_micro_for_no_grad]
				
				prob_old_batch = get_per_token_log_probs(
					model_policy,  # 使用当前policy作为old policy
					input_ids=sequence_ids_batch,
					attention_mask=sequence_mask_batch,
				)
				prob_ref_batch = get_per_token_log_probs(
					model_reference,
					input_ids=sequence_ids_batch,
					attention_mask=sequence_mask_batch,
				)
				
				prob_per_token_old.append(prob_old_batch)
				prob_per_token_reference.append(prob_ref_batch)
			
			# 将mini batch结果拼接
			prob_per_token_old = torch.cat(prob_per_token_old, dim=0)
			prob_per_token_reference = torch.cat(prob_per_token_reference, dim=0)

		loss_list = []
		
		for _ in range(mu):

			optimizer.zero_grad()
			for i in range(0, len(sequence_ids), batch_size_micro):
				
				sequence_ids_batch = sequence_ids[i:i + batch_size_micro]
				sequence_mask_batch = sequence_mask[i:i + batch_size_micro]
				completion_mask_batch = completion_mask[i:i + batch_size_micro]
				group_advantage_per_sample_batch = group_advantage_per_sample[i:i + batch_size_micro]

				# 使用预先计算的固定old_policy_prob和reference_prob
				prob_per_token_old_batch = prob_per_token_old[i:i + batch_size_micro]
				prob_per_token_reference_batch = prob_per_token_reference[i:i + batch_size_micro]

				tmp = []
					
				loss = get_grpo_loss(
					model_policy,
					sequence_ids_batch,
					sequence_mask_batch,
					completion_mask_batch,
					group_advantage_per_sample_batch,
					prob_per_token_old_batch,
					prob_per_token_reference_batch,
					epsilon,
					beta
				)
				loss.backward()
			optimizer.step()

			tmp.append(loss.item())
				
			loss_list.append(np.mean(tmp).item())

		rewards = {
			k: v.item() 
			for k, v in zip(
				[item.__name__ for item in reward_funcs], 
				reward_per_reward_func
			)
		}
		rewards['mean_reward'] = np.mean(list(rewards.values())).item()
		# for logging
		log_info = {
			'epoch': epoch,
			'step': step,
			'loss': loss_list,
			'learning_rate': scheduler.get_last_lr()[0],
			**rewards
		}
		step += 1

		scheduler.step()  # 更新学习率
		tqdm.write(str(log_info))
		wandb.log(log_info)

		# break
	
	# break

# 保存训练完成的模型
model_save_path = f"./grpo_{model_name_or_path.split('/')[-1]}_{start_time}_epoch_{n_epoch}"
model_policy.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

wandb.finish()