# BERT-Based Review Informativeness Classifier

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# Custom Dataset Class
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

def load_and_prepare_data():
    """Load and prepare the informativeness dataset"""
    print("Loading dataset...")
    
    try:
        # Load the informativeness dataset
        df = pd.read_csv('dataset/train_informative.csv')
        print(f"✅ Loaded dataset with {len(df)} reviews")
    except FileNotFoundError:
        print("❌ dataset/test_informative.csv not found. Creating sample data...")
        # Create sample data for demonstration
        sample_data = {
            'review_text': [
                "Great food!",
                "The service was excellent and the staff was very friendly. The atmosphere was cozy.",
                "Terrible place, worst experience ever.",
                "I visited on Tuesday evening around 7 PM. The wait was about 15 minutes. Our server Mike was attentive and knowledgeable. I ordered the salmon dinner ($24) which was perfectly cooked. The portion size was generous and the vegetables were fresh.",
                "Amazing restaurant with incredible attention to detail. The chef personally explained each dish.",
                "OK place.",
                "The food quality has declined significantly since my last visit 6 months ago. The chicken was overcooked and the service was slow despite the restaurant being half empty. I waited 20 minutes just to get the check.",
                "Love it!",
                "Decent restaurant with reasonable prices. The pasta was good but nothing special."
            ],
            'informativeness_level': [0, 1, 0, 2, 1, 0, 2, 0, 1]
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample dataset with {len(df)} reviews")
    
    # Clean text
    df['review_text'] = df['review_text'].astype(str).str.strip()
    
    
    print(f"Data distribution:")
    print(f"  Informative reviews: {df['is_informative'].sum()} ({df['is_informative'].mean():.1%})")
    print(f"  Non-informative reviews: {(~df['is_informative'].astype(bool)).sum()} ({1-df['is_informative'].mean():.1%})")
    
    return df

def create_balanced_dataset(df, max_samples_per_class=1000):
    """Create a balanced dataset for training"""
    
    informative_reviews = df[df['is_informative'] == 1]
    non_informative_reviews = df[df['is_informative'] == 0]
    
    print(f"Original distribution:")
    print(f"  Informative: {len(informative_reviews)}")
    print(f"  Non-informative: {len(non_informative_reviews)}")
    
    # Balance the dataset
    min_samples = min(len(informative_reviews), len(non_informative_reviews))
    samples_per_class = min(min_samples, max_samples_per_class)
    
    balanced_informative = informative_reviews.sample(n=samples_per_class, random_state=42)
    balanced_non_informative = non_informative_reviews.sample(n=samples_per_class, random_state=42)
    
    balanced_df = pd.concat([balanced_informative, balanced_non_informative]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset:")
    print(f"  Total samples: {len(balanced_df)}")
    print(f"  Informative: {balanced_df['is_informative'].sum()}")
    print(f"  Non-informative: {len(balanced_df) - balanced_df['is_informative'].sum()}")
    
    return balanced_df

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {'accuracy': accuracy}

def train_bert_model(df):
    """Train BERT model for informativeness classification"""
    
    # Prepare data
    X = df['review_text'].values
    y = df['is_informative'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Data splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Initialize tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Move model to device
    model.to(device)
    
    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    val_dataset = ReviewDataset(X_val, y_val, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./bert_informativeness_model',
        num_train_epochs=2,  # Reduced from 3 to 2 for faster training
        per_device_train_batch_size=4,  # Reduced from 8 to 4 to handle memory issues
        per_device_eval_batch_size=4,   # Reduced from 8 to 4 to handle memory issues
        warmup_steps=100,  # Reduced from 500
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # More frequent logging
        eval_strategy='steps',
        eval_steps=50,     # More frequent evaluation
        save_strategy='steps',
        save_steps=50,     # More frequent saving
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        report_to=None,
        seed=42,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        fp16=False,  # Disable mixed precision to avoid potential issues
        dataloader_num_workers=0  # Disable multiprocessing
    )
    
    # Create trainer
    print("📝 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model with error handling
    print("\n🚀 Starting BERT training...")
    try:
        training_output = trainer.train()
        print("✅ Training completed successfully!")
        print(f"Training output: {training_output}")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    try:
        test_results = trainer.evaluate(test_dataset)
        print(f"✅ Evaluation completed: {test_results}")
    except Exception as e:
        print(f"❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Get detailed predictions
    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    
    # Print results
    print(f"\n✅ Training completed!")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Informative', 'Informative']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Informative', 'Informative'],
                yticklabels=['Not Informative', 'Informative'])
    plt.title('Confusion Matrix - BERT Informativeness Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Save model
    model_save_path = './bert_informativeness_final'
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    
    return trainer, tokenizer, (X_test, y_test, y_pred)

def test_model_predictions(trainer, tokenizer):
    """Test the model with sample predictions"""
    
    test_reviews = [
        "Great food!",  # Not informative
        "The service was excellent and the staff was very friendly. The atmosphere was cozy and the food was delicious.",  # Moderately informative
        "I visited last Tuesday around 7 PM. The wait time was about 15 minutes which was reasonable. Our server, Mike, was attentive and knowledgeable about the menu. I ordered the salmon dinner ($24) and my partner had the chicken special ($18). Both dishes were well-prepared and came with fresh vegetables.",  # Very informative
        "Terrible experience. Worst place ever.",  # Not informative
        "The restaurant has a nice ambiance with dim lighting and comfortable seating. The menu offers a good variety of dishes ranging from $15-30. Our appetizer arrived within 10 minutes and was beautifully presented."  # Informative
    ]
    
    print("\n🧪 Testing Model Predictions:")
    print("=" * 80)
    
    for i, text in enumerate(test_reviews, 1):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            confidence = probabilities.max().item()
        
        pred_label = 'Informative' if prediction.item() == 1 else 'Not Informative'
        
        print(f"\nReview {i}:")
        print(f"Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"Prediction: {pred_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities: Not Informative: {probabilities[0][0]:.3f}, Informative: {probabilities[0][1]:.3f}")
        print("-" * 60)

def main():
    """Main function to run the BERT training pipeline"""
    
    print("🤖 BERT Review Informativeness Classifier")
    print("=" * 50)
    
    try:
        # Load and prepare data
        print("📂 Loading and preparing data...")
        df = load_and_prepare_data()
        print(f"✅ Data loaded successfully with {len(df)} samples")
        
        # Create balanced dataset
        print("⚖️ Creating balanced dataset...")
        balanced_df = create_balanced_dataset(df)
        print(f"✅ Balanced dataset created with {len(balanced_df)} samples")
        
        # Train BERT model
        print("🏋️ Starting model training...")
        trainer, tokenizer, test_results = train_bert_model(balanced_df)
        
        # Test with sample predictions
        print("🧪 Testing model predictions...")
        test_model_predictions(trainer, tokenizer)
        
        print("\n✅ Training pipeline completed!")
        print("💾 Model saved and ready for use!")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()