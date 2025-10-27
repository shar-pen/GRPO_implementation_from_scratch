from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader

default_system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>\n reasoning process here \n</think>\n<answer>\n answer here \n</answer>.
"""

def extract_final_answer(text):
	if "####" not in text:
		return None
	return text.split("####")[1].strip()





def make_conversation(example, system_prompt=None):
	prompt = []

	if system_prompt is not None:
		prompt.append({"role": "system", "content": system_prompt})
	
	prompt.append({"role": "user", "content": example['question']})

	return {"prompt": prompt, "solution": extract_final_answer(example['answer'])}

def add_len(example, tokenizer):
    # 计算 token 数；去掉 special tokens 保持一致性
    prompt_ids  = tokenizer.apply_chat_template(example["prompt"], tokenize=True, add_generation_prompt=True)
    answer_ids  = tokenizer.encode(example["answer"],  add_special_tokens=False)
    example["prompt_len"]  = len(prompt_ids)
    example["answer_len"]  = len(answer_ids)
    return example

def dummy_data_collator(features):
	return features


def prepare_dataloader(tokenizer, system_prompt=default_system_prompt, batch_size=1):

	dataset = load_dataset('openai/gsm8k', 'main', split='train')

	dataset_formatted = dataset.map(
		partial(
			make_conversation, 
			system_prompt=system_prompt,
		),
	)
	dataset_formatted = dataset_formatted.map(
		partial(add_len, tokenizer=tokenizer),
	)

	dataset_formatted = dataset_formatted.filter(
		lambda x: x["prompt_len"] <= 300 and x["answer_len"] <= 200,
	)
	dataset_formatted = dataset_formatted.select(range(1024))

	dataset_formatted_split = dataset_formatted.train_test_split(test_size=0.1)
	train_dataset = dataset_formatted_split['train']
	eval_dataset = dataset_formatted_split['test']

	dataloader_params = {
		"batch_size": batch_size,
		"collate_fn": dummy_data_collator, 
		"pin_memory": False,
		"drop_last": True,
	}

	train_dataloader = DataLoader(train_dataset, **dataloader_params)
	eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
	
	return train_dataloader, eval_dataloader


def maybe_apply_chat_template(prompt, tokenizer):
	"""
	inspried by trl's maybe_apply_chat_template
	"""
	if isinstance(prompt, str):
		prompt = [{"role": "user", "content": prompt}]
	elif isinstance(prompt, list):
		pass
	else:
		raise ValueError("Prompt must be a string or a list of dictionaries.")

	return tokenizer.apply_chat_template(
		prompt, 
		tokenize=False,
		add_generation_prompt=True,
	)


if __name__ == "__main__":
	
	from transformers import AutoTokenizer
	model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	train_loader, eval_loader = prepare_dataloader(tokenizer)
	print(f"Train loader size: {len(train_loader)}")
	print(f"Eval loader size: {len(eval_loader)}")
	for batch in train_loader:
		print(batch)
		break  # Just to show one batch