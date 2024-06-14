import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


def createDataset(dataset_path):
    df = pd.read_json(dataset_path, lines=True)
    df = df.drop(columns=["subject"])
    df = df.rename(columns={"question": "prompt", "answer": "completion"})
    df_small = df.head(3)
    dataset = Dataset.from_pandas(df_small)
    return dataset


config = AutoConfig.from_pretrained(
    "./checkpoints/mcq_hf_model", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/mcq_hf_model", config=config, trust_remote_code=True
)

DATASET_PATH_EVAL = "./data/MMLU_mcq_clean_test.jsonl"
DATASET_PATH_TRAIN = "./data/MMLU_mcq_clean_train.jsonl"

dataset_train = createDataset(DATASET_PATH_TRAIN)
dataset_eval = createDataset(DATASET_PATH_EVAL)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512


print(model)

print("===================")
print(dataset_train[0:2]["prompt"])
print("===================")

inputs = tokenizer(
    dataset_train[0:2]["prompt"],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512,
)


def convert_to_class(prediction_letter):
    if prediction_letter == "A":
        return 0
    elif prediction_letter == "B":
        return 1
    elif prediction_letter == "C":
        return 2
    elif prediction_letter == "D":
        return 3


labels = torch.tensor(
    [convert_to_class(entry) for entry in dataset_train[0:2]["completion"]]
)


print("label: ", labels)
print("inputs: ", inputs)

output = model(**inputs, labels=labels)
print(output)
