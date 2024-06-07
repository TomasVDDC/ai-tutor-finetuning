

# from transformers import AutoModelForCausalLM, AutoTokenizer


# def generate_reponses(model,tokenizer):
#     prompt = [
#         "Tell me about the history of the United States of America",
#         "Tell me about the history of the United States of America",
#     ]

#     # , "Tell me about Michelle Obama and her family"

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=12,
#         # padding_side="left",
#     )

#     #print("======================== \n inputs: \n", inputs)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=20,
#     )

#    # print(outputs)

#     output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     #print(output_text)
#     return output_text



# tokenizer_policy = AutoTokenizer.from_pretrained("./checkpoints/dpo_trained_model/new")
# model_policy = AutoModelForCausalLM.from_pretrained(
#     "./checkpoints/dpo_trained_model/new", trust_remote_code=True
# )
# tokenizer_reference = AutoTokenizer.from_pretrained("./checkpoints/initial_model")
# model_reference = AutoModelForCausalLM.from_pretrained(
#     "./checkpoints/initial_model", trust_remote_code=True
# )

# tokenizer_policy.pad_token_id = tokenizer_policy.eos_token_id
# tokenizer_policy.pad_token = tokenizer_policy.eos_token

# tokenizer_reference.pad_token_id = tokenizer_reference.eos_token_id
# tokenizer_reference.pad_token = tokenizer_reference.eos_token


# reponse_policy = generate_reponses(model=model_policy,tokenizer=tokenizer_policy)
# reponse_reference = generate_reponses(model=model_reference,tokenizer=tokenizer_reference)

# print("reponse_policy\n",reponse_policy)

# print("reponse_reference\n",reponse_reference)


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./checkpoints/dpo_trained_model/new", trust_remote_code=True)
model_ref = AutoModelForCausalLM.from_pretrained("./checkpoints/initial_model", trust_remote_code=True)

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

evaluate_dataset_path = "./data/fake_eval_data.jsonl"
train_dataset_path = "./data/fake_training_data.jsonl"

df = pd.read_json(evaluate_dataset_path, lines=True)
eval_dataset = Dataset.from_pandas(df)

df = pd.read_json(train_dataset_path, lines=True)
train_dataset = Dataset.from_pandas(df)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",  
    logging_steps=1,  # Print logs every 50 steps
    logging_dir="./logs",  # Directory to save logs
    save_steps=1000,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear",
    warmup_steps=500,
    report_to="all",  # Reporft to all available integrations (e.g., TensorBoard, stdout)
)

dpo_trainer = DPOTrainer(
    args=training_args,
    model=model,
    ref_model = model_ref,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,   
    # args=training_args,
)

training_results = dpo_trainer.train()
print("Training completed:", training_results) 

eval_dict = dpo_trainer.evaluate()
print("Evaluation completed:",eval_dict)