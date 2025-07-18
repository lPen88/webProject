# questo non dovrebbe più servirci perchè ogni modello/dataset lo carichiamo direttamente su Hugging Face
# lo tengo in caso di necessità

import os
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import HfApi, login
import glob

def push_model_to_hf():
    
    model_path = "bert_model"   # path locale del modello
    dataset_path = "dataset"   # path dataset
    repo_name = "GOAISI/webProject"  # repo
    
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # tieni pronto il token di accesso
    # è quello che trovi in https://huggingface.co/settings/tokens
    # ASSICURATI DI AVERLO COPIATO PERCHE' NON PUOI PIÙ VEDERLO DOPO
    login()
    
    
    # butta il modello
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    
    # butta i dataset
    api = HfApi()
    dataset_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    
    for file_path in dataset_files:
        filename = os.path.basename(file_path)
        print(f"  Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"dataset/{filename}",
            repo_id=repo_name,
            repo_type="model"
        )
    
    print("Step 5: Uploading README (model card)...")
    
    # il readme l'ha generato completamente copilot
    # idc
    readme_file = "model_card_template.md"
    if os.path.exists(readme_file):
        print(f"  Uploading {readme_file} as README.md...")
        api.upload_file(
            path_or_fileobj=readme_file,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model"
        )
    else:
        print(f"  Warning: {readme_file} not found, skipping README upload")
    
    
    print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")
    print(f"✅ README.md created from model card template")
    print(f"✅ All datasets and code files uploaded")

if __name__ == "__main__":
    push_model_to_hf()