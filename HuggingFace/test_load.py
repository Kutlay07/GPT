from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "../huggingface_model",
    trust_remote_code=True
)

print("AutoModel works!")
print(type(model))