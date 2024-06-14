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


class MCQModel(nn.Module):
    def __init__(self, name_model):
        super(MCQModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            name_model,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.classifier = nn.Linear(
            self.model.config.model_dim, 4
        )  # 4 classes for 'A', 'B', 'C', 'D'

    def forward(self, input_ids, attention_mask=None, labels=None):
        print(labels)
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
            loss = loss_fct(
                logits, labels
            )  # labels [batch_size], [logits batch_size x num_classes]
            # print("LOSS", loss)
            outputs["loss"] = loss

        # print("OUTPUTS KEY" ,outputs.keys())
        return outputs


### TO DO add to custom config the auto map so that the model can be loaded with from_pretrained and you don't have to use register
class MyCustomConfig(PretrainedConfig):
    model_type = "mcq_hf_model"

    def __init__(self, name_model="apple/OpenELM-450M-Instruct", **kwargs):
        super().__init__(**kwargs)
        self.name_model = name_model


class MCQHFModel(PreTrainedModel):
    config_class = MyCustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MCQModel(config.name_model)

    def forward(self, x):
        return self.model(x)


# # Save the configuration
# config = MyCustomConfig(name_model="apple/OpenELM-450M-Instruct")
# config.save_pretrained("mcq_hf_model")

# # Save the model
# model = MCQHFModel(config)
# model.save_pretrained("mcq_hf_model")


# # Register the custom model and configuration
# AutoConfig.register("mcq_hf_model", MyCustomConfig)
# AutoModel.register(MyCustomConfig, MCQHFModel)

# # Load configuration and model using from_pretrained
# config = AutoConfig.from_pretrained("mcq_hf_model")
# model = AutoModel.from_pretrained("mcq_hf_model", config=config)


# def createDataset(dataset_path):
#     df = pd.read_json(dataset_path, lines=True)
#     df = df.drop(columns=["subject"])
#     df = df.rename(columns={"question": "prompt", "answer": "completion"})
#     df_small = df.head(3)
#     dataset = Dataset.from_pandas(df_small)
#     return dataset


# DATASET_PATH_EVAL = "./data/MMLU_mcq_clean_test.jsonl"
# DATASET_PATH_TRAIN = "./data/MMLU_mcq_clean_train.jsonl"

# dataset_train = createDataset(DATASET_PATH_TRAIN)
# dataset_eval = createDataset(DATASET_PATH_EVAL)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = 512


# print("===================")
# print(dataset_train[0])
# print("===================")

# # Example input
# input_tensor = torch.randn(1, 2048)
# output = model(input_tensor)
# print(output)
