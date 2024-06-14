from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

# Load the dataset
DATASET_PATH_EVAL = "./data/MMLU_mcq_clean_test.jsonl"
DATASET_PATH_TRAIN = "./data/MMLU_mcq_clean_train.jsonl"
MODEL_PATH = "ailieus/NLP_milestone2"


def createDataset(dataset_path):
    df = pd.read_json(dataset_path, lines=True)
    df = df.drop(columns=["subject"])
    df = df.rename(columns={"question": "prompt", "answer": "completion"})
    df_small = df.head(3)
    dataset = Dataset.from_pandas(df_small)
    return dataset


dataset_train = createDataset(DATASET_PATH_TRAIN)
dataset_eval = createDataset(DATASET_PATH_EVAL)
print("dataset_train", dataset_train)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=1,
    # warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="all",
)

config = AutoConfig.from_pretrained("./mcq_hf_model", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "./mcq_hf_model", config=config, trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "attn.qkv_proj",
        "attn.out_proj",
        "ffn.proj_1",
        "ffn.proj_2",
        "classifier",  # our added classifier head
    ],  # Adjust based on your model's architecture
)

print(model)

model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
)

trainer.train()

print("Training finished")

save_directory = "./checkpoints/mcqa_hf_trained/"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Saving finished")
