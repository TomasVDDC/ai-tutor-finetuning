from training_mcqa import MCQModel, MCQModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch.nn.functional as F
import torch

MODEL_PATH = "./model/checkpoints/mcqa_model_trained_v1"  
DATASET_PATH_TRAIN = "./model/data/MMLU_mcq_clean_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

df_train = pd.read_json(DATASET_PATH_TRAIN, lines=True)
df_train = df_train.head(3)
train_tokenized = tokenizer(list(df_train['question']), padding=True, truncation=True, max_length=512)

print(train_tokenized)

input_ids = torch.tensor(train_tokenized.input_ids)
attention_mask = torch.tensor(train_tokenized.attention_mask)

model = MCQModel.from_pretrained(MODEL_PATH)

# print(model)

out = model(input_ids=input_ids, attention_mask=attention_mask)

probs = F.softmax(out.logits, dim=-1)



print(probs)