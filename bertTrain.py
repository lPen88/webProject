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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
        
        # Tokenize text
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
    """questa bilancia il dataset"""
    
    informative_reviews = df[df['is_informative'] == 1]
    non_informative_reviews = df[df['is_informative'] == 0]
    
    print(f"originale:")
    print(f"    Informative: {len(informative_reviews)}")
    print(f"    Non-informative: {len(non_informative_reviews)}")
    
    # prendo la classe che tiene meno campioni e uso quella come base per il bilanciamento
    min_samples = min(len(informative_reviews), len(non_informative_reviews))
    
    balanced_informative = informative_reviews.sample(n=min_samples, random_state=42)
    balanced_non_informative = non_informative_reviews.sample(n=min_samples, random_state=42)
    
    balanced_df = pd.concat([balanced_informative, balanced_non_informative]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"bilanciato:")
    print(f"  Informative: {balanced_df['is_informative'].sum()}")
    print(f"  Non-informative: {len(balanced_df) - balanced_df['is_informative'].sum()}")
    
    return balanced_df

# questo l'ho rubato
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {'accuracy': accuracy}



df = pd.read_csv('dataset/train_informative.csv')
balanced_df = balance_dataset(df)


X = df['review_text'].values
y = df['is_informative'].values


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

model_save_path = './bert_model'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"modello salvato in: {model_save_path}")