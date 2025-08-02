# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from math_verify import parse, verify
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and prep dataset
SYSTEM_PROMPT = """
The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think>\n reasoning process here \n</think>\n<answer>\n answer here \n</answer>.
"""

XML_COT_FORMAT = """\
<think>
{reasoning}
</think>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
	"""Extracts the answer from XML-formatted text."""
	try:
		answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
		return answer
	except IndexError:
		logger.warning("Failed to extract answer from XML format.")
		return ""

def extract_hash_answer(text: str) -> str | None:
	"""Extracts the answer from a hash-formatted string."""
	if "####" not in text:
		return None
	return text.split("####")[1].strip()

# Validate environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
if not MODEL_NAME:
	raise ValueError("MODEL_NAME environment variable is not set.")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/default-GRPO")
if not OUTPUT_DIR:
	raise ValueError("OUTPUT_DIR environment variable is not set.")

RUN_NAME = os.getenv("RUN_NAME", "default-GRPO-gsm8k")
if not RUN_NAME:
	raise ValueError("RUN_NAME environment variable is not set.")

# Configurable one-shot prompting
def get_gsm8k_questions(split="train", use_one_shot=False) -> Dataset:
	"""Loads and prepares the GSM8K dataset with optional one-shot prompting."""
	try:
		data = load_dataset('openai/gsm8k', 'main')[split]
	except Exception as e:
		logger.error(f"Failed to load dataset: {e}")
		raise

	def format_example(x):
		prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
		if use_one_shot:
			prompt.extend([
				{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
				{'role': 'assistant', 'content': XML_COT_FORMAT.format(
					reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
					answer="7 is the largest single-digit prime number."
				)}
			])
		prompt.append({'role': 'user', 'content': x['question']})
		return {'prompt': prompt, 'answer': extract_hash_answer(x['answer'])}

	return data.map(format_example)

dataset = get_gsm8k_questions(use_one_shot=False)

# Reward functions

def math_verify_answer(answer, golden_answer, **kwargs):
	"""Verifies the answer using math_verify."""
	gold_parsed = parse(golden_answer)
	answer_parsed = parse(answer)
	try:
		return verify(gold_parsed, answer_parsed)
	except Exception as e:
		print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
		return False

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
	"""Calculates reward based on correctness of the response."""
	responses = [completion[0]['content'] for completion in completions]
	q = prompts[0][-1]['content']
	extracted_responses = [extract_xml_answer(r) for r in responses]
	logger.info(f"Question:\n{q}\nAnswer:\n{answer[0]}\nResponse:\n{responses[0]}\nExtracted:\n{extracted_responses[0]}")
	return [2.0 if math_verify_answer(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
	"""Calculates reward if the extracted response is a digit."""
	responses = [completion[0]['content'] for completion in completions]
	extracted_responses = [extract_xml_answer(r) for r in responses]
	return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
	"""Calculates reward based on XML formatting."""
	pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$" if strict else r"<think>.*?</think>\s*<answer>.*?</answer>"
	responses = [completion[0]["content"] for completion in completions]
	matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
	return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
	"""Calculates reward based on XML tag counts."""
	contents = [completion[0]["content"] for completion in completions]
	return [count_xml(c) for c in contents]

def count_xml(text) -> float:
	"""Counts XML tags and penalizes extra content."""
	count = 0.0
	if text.count("<think>\n") == 1:
		count += 0.125
	if text.count("\n</think>\n") == 1:
		count += 0.125
	if text.count("\n<answer>\n") == 1:
		count += 0.125
	if text.count("\n</answer>") == 1:
		count += 0.125
	return count

# Model setup
try:
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME,
		torch_dtype=torch.bfloat16,
		# attn_implementation="flash_attention_2",
		# device_map="auto"
	).to("cuda")
	model.config.use_cache = False
	model.gradient_checkpointing_enable()
except Exception as e:
	logger.error(f"Failed to load model: {e}")
	raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# PEFT config (optional)
peft_config = LoraConfig(
	r=16,
	lora_alpha=64,
	target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
	task_type="CAUSAL_LM",
	lora_dropout=0.05,
)

# Training config
training_args = GRPOConfig(
	output_dir=OUTPUT_DIR,
	run_name=RUN_NAME,
	learning_rate=5e-6,
	adam_beta1=0.9,
	adam_beta2=0.99,
	weight_decay=0.1,
	warmup_ratio=0.1,
	lr_scheduler_type='cosine',
	logging_steps=1,
	bf16=True,
	per_device_train_batch_size=1,  # Increased from 1
	gradient_accumulation_steps=4,  # Reduced from 4
	num_generations=16,  # Reduced from 16
	max_prompt_length=256,
	max_completion_length=512,
	num_train_epochs=1,
	save_steps=100,
	save_total_limit=2,
	max_grad_norm=0.1,
	report_to="wandb",
	log_on_each_node=False,
)

# Trainer setup
trainer = GRPOTrainer(
	model=model,
	processing_class=tokenizer,
	reward_funcs=[
		xmlcount_reward_func,
		format_reward_func,  # No need for lambda, just pass the function
		# int_reward_func, # the answer shouldn't be an integer, user should receive a detailed answer rather than a simple answer.
		correctness_reward_func
	],
	args=training_args,
	train_dataset=dataset,
	# peft_config=peft_config  # Uncomment if PEFT is working for you
)

# Train the model
try:
	trainer.train()
except Exception as e:
	logger.error(f"Training failed: {e}")
	raise