from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi
import torch
import os

def authenticate_huggingface(token="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"):
    """Authenticates with Hugging Face."""
    try:
        if token:
            login(token=token)
            print("Hugging Face login successful using provided token.")
        elif "HUGGINGFACE_TOKEN" in os.environ:
            login(token=os.environ["HUGGINGFACE_TOKEN"])
            print("Hugging Face login successful using environment variable.")
        else:
            print("No Hugging Face token provided or found in environment variables. Please provide a token or set the HUGGINGFACE_TOKEN environment variable.")
            return False
        return True
    except Exception as e:
        print(f"Hugging Face login failed: {e}")
        return False

#LOADS THE LLM
def load_model(model_i = "meta-llama/Llama-3.1-8B"):
    if not authenticate_huggingface():
        print("Authentication failed. Model loading aborted.")
        return None, None  # Return None for both if authentication fails

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_i)
        model = AutoModelForCausalLM.from_pretrained(model_i, device_map = "auto")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

#GENERATES THE TEXT USING THE LLM
def generate_text(prompt, tokenizer, model):
    if tokenizer is None or model is None:
        print("Tokenizer or model not loaded. Text generation aborted.")
        return "" # return empty string to prevent errors
    print("INPUT GIVEN")
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    print("FINISHED INPUT STARTING OUTPUTS")
    outputs = model.generate(
        **inputs,
        max_length=32,  # Reduce max_length
        do_sample=True, #use sampling.
        temperature = 0.7, #add temperature.
        top_p = 0.9, #add top_p.
        #num_beams=1,  # Use greedy decoding (or a smaller number of beams)
    )
    print("FINISHED OUTPUTS")
    return tokenizer.decode(outputs[0], skip_special_tokens=True)