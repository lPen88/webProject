from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download, login
import pandas as pd
from sklearn.metrics import classification_report


login()
repo = "GOAISI/webProject"

#questo scarica il csv e lo infila da qualche parte in .cache, restituisce il path
csv_path = hf_hub_download(repo_id=repo, filename="dataset/test_informative.csv")



tokenizer = BertTokenizer.from_pretrained(repo)
model = BertForSequenceClassification.from_pretrained(repo)

def predict_informativeness(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    return predicted_class


df = pd.read_csv(csv_path)


y_true = df['is_informative'].tolist()
y_pred = [predict_informativeness(text) for text in df['review_text']]

report = classification_report(y_true, y_pred, target_names=["Not Informative", "Informative"])
print(report)


#                     precision    recall  f1-score   support 
#           
#   Not Informative        0.88      0.77      0.82       643
#   Informative            0.97      0.98      0.98      4357

