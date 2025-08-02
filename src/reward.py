import re
import torch
from math_verify import parse, verify


def extract_answer(text):
	match = re.search(r'<answer>\n(.*?)\n</answer>', text, re.DOTALL)
	if match:
		return match.group(1).strip()
	return None


def format_reward(completion, **kwargs):
	"""
	检查预测文本是否符合特定格式要求。e.g., <think>\n...\n</think>\n<answer>\n...\n</answer>
	kwargs 参数可以用于传递额外的配置，但在此函数中未使用。
	"""
	pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
	if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
		return 1.0
	else:
		return 0.0
	

def tag_count_reward(completion, **kwargs):
	"""
	检查文本中 <think> 和 <answer> 标签的数量。
	"""
	score = 0.0
	if completion.count("<think>\n") == 1:
		score += 0.25
	if completion.count("\n</think>\n") == 1:
		score += 0.25
	if completion.count("\n<answer>\n") == 1:
		score += 0.25
	if completion.count("\n</answer>") == 1:
		score += 0.25
	return score


def reasoning_steps_reward(completion, **kwargs):

	pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
	matches = re.findall(pattern, completion)
	score = min(1.0, len(matches) / 3)  # 奖励 3 次以上
	return score


def accuracy_reward(completion, solution, **kwargs):
	"""
	计算预测文本与真实答案之间的准确度奖励。
	"""
	full_answer_content = extract_answer(completion)
	if full_answer_content is None:
		return 0.0

	gold_parsed = parse(solution)
	answer_parsed = parse(full_answer_content)

	try:
		score = float(verify(gold_parsed, answer_parsed))
	except Exception as e:
		print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
		return 0.0

	return score
	

def compute_grpo_reward(completions, solutions, reward_funcs, reward_weights=None):

	if reward_weights is None:
		reward_weights = [1.0/len(reward_funcs)] * len(reward_funcs)

	assert len(reward_weights) == len(reward_funcs), "reward_weight and reward_funcs must have the same length"

	rewards_per_sample_per_func = torch.zeros(len(completions), len(reward_funcs))

	for i, (a_completion, a_solution) in enumerate(zip(completions, solutions)):
		for j, reward_func in enumerate(reward_funcs):
			rewards_per_sample_per_func[i, j] = reward_func(a_completion, solution=a_solution)

	reward_weight_tensor = torch.tensor(reward_weights)
	reward_per_completion = (rewards_per_sample_per_func * reward_weight_tensor).sum(dim=1)

	# return avergaed score of different reward functions
	reward_per_reward_func = rewards_per_sample_per_func.mean(dim=0)

	return reward_per_completion, reward_per_reward_func


def compute_group_advantage(reward_per_sample: torch.Tensor, num_generations: int=None, eps: float = 1e-8, scale_rewards: bool = True):
	"""
	基于 reward 计算 advantage
	"""
	if num_generations is None:
		num_generations = reward_per_sample.shape[0]

	# 计算同一个prompt的多次生成的平均奖励和标准差
	mean_grouped_rewards = reward_per_sample.view(-1, num_generations).mean(dim=1)
	std_grouped_rewards = reward_per_sample.view(-1, num_generations).std(dim=1)
	
	# 将 mean 和 std 重复 num_generations 次，以便与 rewards 的形状匹配
	mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
	std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
	group_advantage = reward_per_sample - mean_grouped_rewards
	if scale_rewards:
		group_advantage /= (std_grouped_rewards + eps)

	return group_advantage



if __name__ == "__main__":
	# Example usage
	completions = [
		"<think>\nLet's solve this step by step. Jamie's last name is 'Grey', which has 4 letters. If Bobbie's last name were to be halved, it would be twice the length of Grey, meaning it would be 8 letters long. Therefore, Bobbie’s last name has 8 letters. If Samantha's last name has 3 fewer letters than Bobbie's, we’d subtract 3 from Bobbie's last name length. Hence, Samantha's last name has 5 letters.\n</think>\n<answer>\n5\n</answer>",
		'<think>\nTo solve this problem, let\'s start by identifying the number of letters in each last name:\n\n1. Jamie\'s last name is "Grey," which has 4 letters.\n2. If Bobbie takes two letters off her last name, her last name would be half the length of Jamie\'s name. Since Jamie\'s name has 4 letters, Bobbie\'s new last name would have 4 / 2 = 2 letters.\n3. Bobbie’s last name has 2 letters less than Samantha’s last name. So, Samantha’s last name would have 2 + 2 = 4 letters.\n</think>\n<answer>\nSamantha’s last name has 7 letters.\n</answer>',
		'To solve this problem, let\'s start by identifying the number of letters in each last name:\n\n1. Jamie\'s last name is "Grey," which has 4 letters.\n2. If Bobbie takes two letters off her last name, her last name would be half the length of Jamie\'s name. Since Jamie\'s name has 4 letters, Bobbie\'s new last name would have 4 / 2 = 2 letters.\n3. Bobbie’s last name has 2 letters less than Samantha’s last name. So, Samantha’s last name would have 2 + 2 = 4 letters.\n<answer>\nSamantha’s last name has 7 letters.\n</answer>',

	]
	solutions = ['7', '7']
	reward_funcs = [format_reward, tag_count_reward, accuracy_reward]

	reward_per_sample, reward_per_reward_func = compute_grpo_reward(completions, solutions, reward_funcs)
	print(reward_per_sample)
	print(reward_per_reward_func)
	