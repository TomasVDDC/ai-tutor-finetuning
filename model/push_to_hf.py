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

# config = AutoConfig.from_pretrained(
#     "./model/checkpoints/mcqa_model_trained_3_epochs_lr_1e-4_all_data", trust_remote_code=True
# )

model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/mcqa_model_downsampled_bf16",trust_remote_code=True
)

model.push_to_hub("ailieus/quantized_model_final")
