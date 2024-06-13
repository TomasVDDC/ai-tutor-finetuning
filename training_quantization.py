from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

MODEL_PATH = "./model/checkpoints/mcqa_model_trained_v1"  
DATASET_PATH_TRAIN = "./model/data/MMLU_mcq_clean_train.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
quantization_config = QuantoConfig(weights="int8", activations="float8")
quantized_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0", quantization_config=quantization_config)

quantized_model.save_pretrained("./model/checkpoints/mcqa_model_trained_quantized_v1")
