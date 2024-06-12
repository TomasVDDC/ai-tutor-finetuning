import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig
import pandas as pd 

# Example Trainer: https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
# Custom Dataset : https://huggingface.co/transformers/v3.1.0/custom_datasets.html
# Config OpenELm https://huggingface.co/apple/OpenELM-270M/blob/main/config.json


MODEL_PATH = "ailieus/NLP_milestone2"  
DATASET_PATH_EVAL = "./model/data/MMLU_mcq_clean_test.jsonl"
DATASET_PATH_TRAIN = "./model/data/MMLU_mcq_clean_train.jsonl"

# Define the custom model with a classifier layer

class MCQModelConfig(PretrainedConfig):
    model_type = "mcq_model"
    def __init__(self, model_name_or_path=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path

class MCQModel(PreTrainedModel):
    config_class = MCQModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, trust_remote_code=True, output_hidden_states=True)
        self.classifier = nn.Linear(self.model.config.model_dim, 4)

    def forward(self, input_ids, attention_mask=None, labels=None):
        print("LABELS: ", labels)
        # print(input_ids)
        # print(attention_mask)
        
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # print(outputs.keys())
        # print(outputs.logits.shape)
        # # print(outputs.hidden_states)
        # print(outputs.hidden_states[0].shape)
        # print(outputs.hidden_states[1].shape)
        # print(len(outputs.hidden_states))
    
        # hidden state is a tuple with all the hidden layer outputs from the attention, 
        # We are only interested in the last hidden layer and the last token
        logits = self.classifier(outputs.hidden_states[-1][:, -1, :])  
        outputs.logits = logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels) #labels [batch_size], [logits batch_size x num_classes]
            #print("LOSS", loss)
            outputs["loss"] = loss
        
        print("OUTPUTS KEY: " ,outputs.keys())
        return outputs

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


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    df_train = pd.read_json(DATASET_PATH_TRAIN, lines=True)
    train_tokenized = tokenizer(list(df_train['question']), padding=True, truncation=True, max_length=512)

    df_eval = pd.read_json(DATASET_PATH_EVAL, lines=True)
    df_eval_small = df_eval.head(20)

    eval_tokenized_small = tokenizer(list(df_eval_small['question']), padding=True, truncation=True, max_length=512)
    eval_labels_small = list(df_eval_small["answer"])

    df_train_small = df_train.head(3)
    #print(df_train_small)
    train_tokenized_small = tokenizer(list(df_train_small["question"]), padding=True, truncation=True, max_length=512)
    train_labels_small = list(df_train_small["answer"])

    # Create dataset
    train_dataset = MCQDataset(train_tokenized_small, train_labels_small)
    eval_dataset = MCQDataset(eval_tokenized_small, eval_labels_small)
   
    model = MCQModel(MCQModelConfig(MODEL_PATH))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        learning_rate=1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        #warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        report_to="all",
    )

    # Custom compute metrics function, https://huggingface.co/docs/transformers/v4.15.0/en/internal/trainer_utils
    def compute_metrics(eval_obj):
        # prediction[0] is the logits
        predictions = eval_obj.predictions[0].argmax(-1)
        labels = eval_obj.label_ids
        print("labels: ", labels)
        print("predictions :", predictions)
        correct = (predictions == labels).sum()
        accuracy = correct / labels.shape[0]

        return {"accuracy": accuracy}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    
    model.save_pretrained("./model/checkpoints/mcqa_model_trained_v1")
    tokenizer.save_pretrained("./model/checkpoints/mcqa_model_trained_v1")

    print('DONE TRAINING')


if __name__ == "__main__":
    main()


# class MCQModel(nn.Module):
#     def __init__(self, model_name):
#         super(MCQModel, self).__init__()
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
#         print(self.model.config.model_dim)
#         self.classifier = nn.Linear(self.model.config.model_dim, 4)  # 4 classes for 'A', 'B', 'C', 'D'
    
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.model(input_ids, attention_mask=attention_mask)
#         print(labels)
#         # print(outputs.keys())
#         # print(outputs.logits.shape)
#         # # print(outputs.hidden_states)
#         # print(outputs.hidden_states[0].shape)
#         # print(outputs.hidden_states[1].shape)
#         # print(len(outputs.hidden_states))
        
#         # hidden state is a tuple with all the hidden layer outputs from the attention, 
#         # We are only interested in the last hidden layer and the last token
#         logits = self.classifier(outputs.hidden_states[-1][:, -1, :])  
#         outputs.logits = logits
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits, labels) #labels [batch_size], [logits batch_size x num_classes]
#             #print("LOSS", loss)
#             outputs["loss"] = loss

        
#         #print("OUTPUTS KEY" ,outputs.keys())
#         return outputs
    



    # class MyConfig(PretrainedConfig):
#     def __init__(self, classifier, **kwargs):
#         super().__init__(**kwargs)
#         self.classifier = classifier


# class MCQModel(PreTrainedModel):
#     config_class = MyConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True)
#         print(self.model.config.model_dim)
    
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.model(input_ids, attention_mask=attention_mask)
        
#         # hidden state is a tuple with all the hidden layer outputs from the attention, 
#         # We are only interested in the last hidden layer and the last token
#         logits = self.classifier(outputs.hidden_states[-1][:, -1, :])  
#         outputs.logits = logits
#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits, labels) #labels [batch_size], [logits batch_size x num_classes]
#             #print("LOSS", loss)
#             outputs["loss"] = loss

        
#         #print("OUTPUTS KEY" ,outputs.keys())
#         return outputs