import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig
import pandas as pd 

# Example Trainer: https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# Custom Dataset : https://huggingface.co/transformers/v3.1.0/custom_datasets.html
# Config OpenELm https://huggingface.co/apple/OpenELM-270M/blob/main/config.json


MODEL_PATH = "ailieus/NLP_milestone2"  
DATASET_PATH_EVAL = "./model/data/MMLU_mcq_clean_test.jsonl"
DATASET_PATH_TRAIN = "./model/data/MMLU_mcq_clean_train.jsonl"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

df_train = pd.read_json(DATASET_PATH_TRAIN, lines=True)
train_tokenized = tokenizer(list(df_train['question']), padding=True, truncation=True, max_length=512)

df_eval = pd.read_json(DATASET_PATH_EVAL, lines=True)
eval_tokenized = tokenizer(list(df_eval['question']), padding=True, truncation=True, max_length=512)

def convert_to_class(prediction_letter):
    if prediction_letter == "A":
        return 0
    elif prediction_letter == "B":
        return 1
    elif prediction_letter == "C":
        return 2
    elif prediction_letter == "D":
        return 3

# Define your dataset
class MCQDataset(Dataset):
    def __init__(self, tokenized_data, labels=None):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_data["input_ids"])

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_data.items()}
        if self.labels:
            label_converted = convert_to_class(self.labels[idx])
            item["labels"] = torch.tensor(label_converted)
        return item


df_train_small = df_train.head(3)
print(df_train_small)
train_tokenized_small = tokenizer(list(df_train_small["question"]), padding=True, truncation=True, max_length=512)
train_labels = list(df_train_small["answer"])

# Create dataset
train_dataset = MCQDataset(train_tokenized_small, train_labels)


###################################
# TRAINING MODEL 
###################################

config = AutoConfig.from_pretrained("./model/checkpoints/mcq_hf_model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./model/checkpoints/mcq_hf_model", config=config, trust_remote_code=True
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

model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    #warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Custom compute metrics function
def compute_metrics(pred):
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids
    correct = (predictions == labels).sum()
    accuracy = correct / labels.shape[0]
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

model.save_pretrained("./model/checkpoints/mcqa_model_trained")
tokenizer.save_pretrained("./model/checkpoints/mcqa_model_trained")

print('DONE TRAINING')