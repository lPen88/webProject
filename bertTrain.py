#qui addestro il modellino con BERT
#assicurati di aver eseguito preprocessing.py prima

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from huggingface_hub import hf_hub_download, login
from huggingface_hub import HfApi
import shutil
import os
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

#questo serve a vedere se la GPU è disponibile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Uso: {device}")
print(f"PyTorch version: {torch.__version__}")

# questa classe la tengo per rendere più semplice la gestione del dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # questo viene chiamato in automatico tra trainer
        # o almeno così ho capito
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def balance_dataset(df):
    
    # Prendi 4500 campioni per ciascuna classe (0 e 1)
    # sono 9000 in tutto perchè colab poi vuole che pago (stronzi)
    class_0 = df[df['is_informative'] == 0].sample(n=4500, random_state=42)
    class_1 = df[df['is_informative'] == 1].sample(n=4500, random_state=42)
    balanced_df = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

# questo l'ho rubato
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {'accuracy': accuracy}

repo = "GOAISI/webProject"
login()

#questo scarica il csv e lo infila da qualche parte in .cache, restituisce il path
csv_path = hf_hub_download(repo_id=repo, filename="dataset/train_informative.csv")
df= pd.read_csv(csv_path)
balanced_df = balance_dataset(df)


X = balanced_df['review_text'].values
y = balanced_df['is_informative'].values


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


print(f"  Training: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)


model.to(device)

train_dataset = ReviewDataset(X_train, y_train, tokenizer)
val_dataset = ReviewDataset(X_val, y_val, tokenizer)
test_dataset = ReviewDataset(X_test, y_test, tokenizer)

# qui ci ho diminuito vari parametri perchè si interrompeva senza lanciare errori
# immagino perchè non ho GPU in vm e la cpu fa schifo
training_args = TrainingArguments(
    output_dir='./bert_model_temp',
    num_train_epochs=2,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,   
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_strategy='steps',
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to=None,
    seed=42,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    fp16=False,
    dataloader_num_workers=0
)

print("creazione trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("addestramento")

training_output = trainer.train()
print(f"finito: {training_output}")

test_results = trainer.evaluate(test_dataset)
print(f"valutazione test set: {test_results}")

#lo salvo prima in locale perchè non sono sicuro se trainer è già il modello o se me lo tira fuori save_model(...)
model_save_path = './bert_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"modello salvato in: {model_save_path}")

#lo carico su Hugging Face
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)


model.push_to_hub(repo)
tokenizer.push_to_hub(repo)

if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)
    print(f"Cartella temporanea '{model_save_path}' eliminata.")