# ATTENZIONE 

modello e dataset sono stati caricati su Hugging Face dalla repo "lPen88/webProject" perchè git LFS si paga (stronzi) 

Creati l'account, poi aggiungici una chiave ssh ed un "Access Token" perchè alcune API (tipo per pushare) non usano ssh  

Non sono sicuro di aver cambiato tutti i path nei file, attento se esegui qualcosa. Per colab boh immagino puoi tirare roba da li senza problemi.

---
### Caricare roba da lì

**dataset:** 
```python
import pandas as pd
from huggingface_hub import hf_hub_download

dataset="quelloCheVuoi"
train_file = hf_hub_download(repo_id="lPen88/webProject", filename="dataset/{dataset}.csv")
df = pd.read_csv(train_file)
```  

**modello:**  
```python
from transformers import BertTokenizer, BertForSequenceClassification

model_dir = "lPen88/webProject"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
```