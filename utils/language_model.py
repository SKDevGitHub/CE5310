from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#LOADS THE LLM
def load_model(model_path = "models/llama_weights/"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto")
    return tokenizer, model

#GENERATES THE TEXT USING THE LLM
def generate_text(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)