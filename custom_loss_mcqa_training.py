from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
import torch

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
print("dataset_train" ,dataset_train)

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
    #warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="all",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)

# Define a custom loss function
def custom_loss(outputs, labels):
    logits = outputs.logits
    loss = torch.nn.functional.mse_loss(logits, labels)
    return loss

# Custom Trainer to output the loss
class CustomTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        print("inputs", inputs)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = custom_loss(outputs, labels)
        if return_outputs:
            return loss, outputs
        return loss

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
    ],  # Adjust based on your model's architecture
)

print(model)

# model = get_peft_model(model, lora_config)

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_train,
#     eval_dataset=dataset_eval,
#     tokenizer=tokenizer,
# )

# trainer.train()

# print("Training finished")

# save_directory = "./checkpoints/mcqa_custom_loss_training/"

# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

# print("Saving finished")


