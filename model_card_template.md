# BERT Review Classifier

This repository contains a fine-tuned BERT model for classifying reviews as informative or not informative, along with the training and test datasets.

## 📁 Repository Contents

### Model Files
- **Model**: Fine-tuned BERT for sequence classification
- **Tokenizer**: BERT tokenizer configuration
- **Config**: Model configuration files

### Dataset Files
- `dataset/train_informative.csv` - Training dataset with informative labels
- `dataset/train_informativeTOT.csv` - Complete training dataset
- `dataset/test_informative.csv` - Test dataset with informative labels
- `dataset/train.csv` - Original training data
- `dataset/test.csv` - Original test data

## 🤖 Model Description

- **Base Model**: BERT (bert-base-uncased)
- **Task**: Binary text classification
- **Dataset**: Review text classification dataset
- **Classes**: 
  - 0: Not informative
  - 1: Informative

## 🚀 Quick Start

### Load the Model

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('lPen88/webProject')
model = BertForSequenceClassification.from_pretrained('lPen88/webProject')

# Example usage
text = "This product is amazing and works exactly as described!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    
print(f"Prediction: {'Informative' if predictions.item() == 1 else 'Not Informative'}")
```

### Load the Datasets

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# Download dataset files
train_file = hf_hub_download(repo_id="lPen88/webProject", filename="dataset/train_informative.csv")
test_file = hf_hub_download(repo_id="lPen88/webProject", filename="dataset/test_informative.csv")

# Load datasets
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
```

## 📊 Dataset Information

The dataset contains review text with binary labels indicating whether the review is informative or not.

### Features
- **review_text**: The text content of the review
- **is_informative**: Binary label (0 = not informative, 1 = informative)

### Statistics
- Training set: Available in multiple variants
- Test set: Used for model evaluation
- Text length: Up to 512 tokens (BERT limit)

## 🔧 Training Details

- Fine-tuned on review classification dataset
- Optimized for identifying informative reviews
- Input length: up to 512 tokens
- Base model: BERT (bert-base-uncased)

## 📈 Performance

The model has been evaluated on the test dataset. See `bertTest.py` for evaluation metrics.

## 💡 Intended Use

This model is designed to classify review text to determine if it contains informative content. It can be used for:

- Content moderation
- Review quality assessment
- Information filtering
- Text classification tasks

## 🔄 Reproducing Results

1. Download the datasets from this repository
2. Use the provided training script: `bertTrain.py`
3. Evaluate using: `bertTest.py`

## 📝 Citation

If you use this model or dataset, please cite appropriately.

## 📄 License

Please check the license requirements for the base BERT model and datasets.
