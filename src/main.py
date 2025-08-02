r"""
This file is old code for GRPO training. I stop using it because it requires too much GPU memory. 
The newer code is in `src/main_minibatch.py`, which uses mini-batch training to reduce memory usage. 
This file maybe contain some implementation errors, so I don't recommend using it.
It is kept here for reference only.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json, re, torch, copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from prepare_data import prepare_dataloader
from reward import format_reward, tag_count_reward, accuracy_reward, compute_grpo_reward, compute_group_advantage
from grpo_utils import generate_rollouts, get_per_token_log_probs, get_grpo_loss
from model_utils import optimize_model_settings, freeze_model


n_epoch = 3
n_roullout = 4
max_length = 500
batch_size_macro = 4 # num of data for update per step
batch_size_dataloader = batch_size_macro // n_roullout # num of batch per step
batch_size_micro = 2 # num of data for backward
learning_rate = 1e-6
epsilon = 0.2 # PPO ratio clipping threshold
beta = 0.04 # KL divergence coefficient
mu = 4 # num of policy updates per batch



train_dataloader, eval_dataloader = prepare_dataloader(batch_size=batch_size_dataloader)
reward_funcs = [format_reward, tag_count_reward, accuracy_reward]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model_policy = AutoModelForCausalLM.from_pretrained(
	model_name_or_path,
	torch_dtype='auto',
).to(device)
optimize_model_settings(model_policy)



logs = []
optimizer = torch.optim.AdamW(model_policy.parameters(), lr=learning_rate)


for epoch in range(n_epoch):

	print(f"Epoch {epoch + 1}/{n_epoch}")

	model_reference = copy.deepcopy(model_policy)
	freeze_model(model_reference)

	for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader)):

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

		# rollout 后计算 old_policy_prob，反向传播时不再重新计算 old_policy_prob
		with torch.no_grad():
			prob_per_token_old = get_per_token_log_probs(
				model_policy,
				input_ids=sequence_ids,
				attention_mask=sequence_mask,
			)
			prob_per_token_reference = get_per_token_log_probs(
				model_reference,
				input_ids=sequence_ids,
				attention_mask=sequence_mask,
			)

		# 将 solutions 扩展到与 completions 相同
		solutions = [s for s in solutions for _ in range(n_roullout)]

		reward_per_completion, reward_per_reward_func = compute_grpo_reward(
			completions, 
			solutions, 
			reward_funcs
		)

		group_advantage_per_sample = compute_group_advantage(
			reward_per_completion
		).to(device)

		for _ in range(mu):
			
			optimizer.zero_grad()

			loss = get_grpo_loss(
				model_policy,
				sequence_ids,
				sequence_mask,
				completion_mask,
				group_advantage_per_sample,
				prob_per_token_old,
				prob_per_token_reference,
				epsilon,
				beta
			)

			# for logging
			logs.append({
				'loss': loss.item(),
				**{
					k: v.item() 
					for k, v in zip(
						[item.__name__ for item in reward_funcs], 
						reward_per_reward_func
					)
				}
			})

			loss.backward()
			optimizer.step()

		# break
	
	# break

# save logs
with open('logs.json', 'w') as f:
	json.dump(logs, f, indent=2)