from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download, login
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Authenticate with Hugging Face Hub
login()

# Download the test CSV from the repo
csv_path = hf_hub_download(repo_id="GOAISI/webProject", filename="dataset/test_informative.csv")

# Load model and tokenizer from Hugging Face Hub
model_dir = "GOAISI/webProject"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

def predict_informativeness(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    return predicted_class

# Load the test CSV
df = pd.read_csv(csv_path)


y_true = df['is_informative'].tolist()  # Replace with your actual label column name
y_pred = [predict_informativeness(text) for text in df['review_text']]

report = classification_report(y_true, y_pred, target_names=["Not Informative", "Informative"])
print(report)


#                     precision    recall  f1-score   support 
#           
#   Not Informative        0.88      0.77      0.82       643
#   Informative            0.97      0.98      0.98      4357

