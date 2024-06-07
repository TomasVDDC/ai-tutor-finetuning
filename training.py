from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

## TODO:
#1. Add eval_dataset currently just a copy of the fake data
#2. If a bigger dataset is added change the logging 

data_train_path = "./data/train_data_cleaned.jsonl"
data_test_path = "./data/test_data_cleaned.jsonl"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-450M-Instruct", trust_remote_code=True
)

# print("====================================================")
# print(model)
# print("====================================================")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "attn.qkv_proj",
        "attn.out_proj",
    ],  # Adjust based on your model's architecture
)

model = get_peft_model(model, lora_config)

model_ref = AutoModelForCausalLM.from_pretrained(
    "apple/OpenELM-450M-Instruct", trust_remote_code=True
)

df = pd.read_json(data_train_path, lines=True)
train_dataset = Dataset.from_pandas(df)
train_dataset = train_dataset.shuffle(seed=42)

df = pd.read_json(data_test_path, lines=True)
eval_dataset = Dataset.from_pandas(df)
eval_dataset = eval_dataset.shuffle(seed=42)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="no",  # Disable evaluation
    logging_steps=2,  # Print logs every 50 steps
    logging_dir="./logs",  # Directory to save logs
    save_steps=1000,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear",
    warmup_steps=500,
    report_to="all",  # Report to all available integrations (e.g., TensorBoard, stdout)
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()

save_directory = "./checkpoints/dpo_trained_model/"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print('DONE TRAINING')


# Early Example of generating with the model

# from transformers import AutoModelForCausalLM, AutoTokenizer

# print("HELLO WORLD")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# model = AutoModelForCausalLM.from_pretrained(
#     "apple/OpenELM-450M-Instruct", trust_remote_code=True
# )

# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token = tokenizer.eos_token

# prompt = [
#     "Tell me about the history of the United States of America",
#     "Tell me about the history of the United States of America",
# ]

# # , "Tell me about Michelle Obama and her family"

# inputs = tokenizer(
#     prompt,
#     return_tensors="pt",
#     padding="max_length",
#     truncation=True,
#     max_length=12,
#     # padding_side="left",
# )

# print("======================== \n inputs: \n", inputs)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=20,
# )

# print(outputs)

# output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(output_text)

# save_path = "./checkpoints/initial_model/"
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)