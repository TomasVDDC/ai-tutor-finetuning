from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig, AutoConfig
# from optimum.quanto import qint8,quantize, freeze
from quanto import quantize
import torch
from peft import PeftModel, PeftConfig
from copy import deepcopy

def print_param_dtype(model):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")

PEFT_CONFIG_PATH = "./checkpoints/mcqa_model_trained_3_epochs_quantized"  
DATASET_PATH_TRAIN = "./data/MMLU_mcq_clean_train.jsonl"

peft_config = PeftConfig.from_pretrained(PEFT_CONFIG_PATH)
print("Adapter Names:", peft_config)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path, trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
   "./checkpoints/mcq_hf_model", trust_remote_code=True,
)


model_bf16 = deepcopy(base_model)
model_bf16 = model_bf16.to(torch.bfloat16)
print_param_dtype(model_bf16)



model_bf16.save_pretrained("./checkpoints/mcqa_model_downsampled_bf16")
