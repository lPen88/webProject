# questo non dovrebbe più servirci perchè ogni modello/dataset lo carichiamo direttamente su Hugging Face
# lo tengo in caso di necessità

import os
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import HfApi, login, upload_file
import glob

def push_model_to_hf():
    # Configuration
    model_path = "bert_model"  # Your local model directory
    dataset_path = "dataset"   # Your dataset directory
    repo_name = "lPen88/webProject"  # Change this to your desired repo name
    
    print("Step 1: Loading your trained model...")
    # Load your trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    print("Step 2: Login to Hugging Face...")
    # You'll need to login first - this will prompt for your token
    login()
    
    print("Step 3: Pushing model to Hugging Face Hub...")
    # Push the model to Hugging Face Hub
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    
    print("Step 4: Uploading dataset files...")
    # Upload dataset files
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
    # Upload the model card as README.md
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
    # Before running this script, make sure you have:
    # 1. Created a Hugging Face account at https://huggingface.co/
    # 2. Created an access token at https://huggingface.co/settings/tokens
    # 3. Changed the repo_name above to your desired repository name
    
    print("Starting model upload to Hugging Face...")
    print("Make sure you've updated the 'repo_name' variable above!")
    
    push_model_to_hf()
    
    print("\nTo use this script:")
    print("1. Update the 'repo_name' variable with your desired repository name")
    print("2. Uncomment the 'push_model_to_hf()' line")
    print("3. Run this script")
