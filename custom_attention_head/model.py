from .configuration_openelm import OpenELMConfig
from modeling_openelm import OpenELMForCausalLM
from transformers import AutoModelForCausalLM
from torch import nn


class OpenELMCustomHeadForCausalLM(OpenELMForCausalLM):
    def __init__(self, config: OpenELMConfig):
        super().__init__(config)

        self.model = AutoModelForCausalLM.from_pretrained(
            "apple/OpenELM-450M-Instruct",
            trust_remote_code=True,
            output_hidden_states=True,
        )

        self.classifier = nn.Linear(self.model.config.model_dim, 4)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        print(labels)
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
