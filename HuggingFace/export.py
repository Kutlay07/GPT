import torch
import shutil
import os

from configuration_gpt_fs import GPTFromScratchConfig
from modeling_gpt_fs import GPTFromScratchForCausalLM


CHECKPOINT_PATH = r"C:\Users\W11\Desktop\GPT\checkpoints\checkpoint_loss_2.28.pt"
EXPORT_PATH = "../huggingface_model"


def export_model():

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    config = GPTFromScratchConfig()

    model = GPTFromScratchForCausalLM(config)


    print("Loading weights...")

    state_dict = checkpoint["model_state_dict"]
    
    hf_state_dict = {}
    
    for key,value in state_dict.items():
        hf_state_dict["transformer."+key] = value
        
    model.load_state_dict(hf_state_dict)
    
    model.tie_weights()

    model.eval()
    
    config.auto_map = {
    "AutoConfig": "configuration_gpt_fs.GPTFromScratchConfig",
    "AutoModelForCausalLM": "modeling_gpt_fs.GPTFromScratchForCausalLM"
    }

    print("Saving Hugging Face model...")

    config.save_pretrained(
        EXPORT_PATH
    )
    
    model.save_pretrained(
        EXPORT_PATH
    )

# Copy HF custom code files
    shutil.copy(
        "configuration_gpt_fs.py",
        os.path.join(EXPORT_PATH, "configuration_gpt_fs.py")
    )

    shutil.copy(
        "modeling_gpt_fs.py",
        os.path.join(EXPORT_PATH, "modeling_gpt_fs.py")
    )



    print("Export completed!")
    print(f"Saved to: {EXPORT_PATH}")


if __name__ == "__main__":
    export_model()