# qui praticamente prendo il df e vedo di ricavare se ogni review è informativa o meno

# in pratica per capire se una review è informtiva o meno lo facciamo in base a quattro cose:
# 1. Lunghezza della review
# 2. Quanto è specifica la review (se va nel dettaglio di roba)
# 3. Quanto è facile da leggere (Formula di Flesch)
# 4. Ratio di parole emotive e fattuali

# do un punteggio in base a queste cose e poi le metto tutte insieme per fare una classificazione
# se lo faccio uscire tra 0 e 1 sono un grande


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from textstat import flesch_reading_ease
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from huggingface_hub import hf_hub_download, login
from huggingface_hub import HfApi
import shutil
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

repo = "GOAISI/webProject"
login()



train_file = hf_hub_download(repo_id=repo, filename="dataset/train.csv")
df = pd.read_csv(train_file)

df = df.head(1000).copy()

print(df.shape)
print(df.columns.tolist())
print(df.head(1))

# "class index" non mi importa perchè è la classificazione di se è positiva o meno (idc)
df = df.drop(columns=['class_index'])

##############################################
############ PREPROCESSING ###################
##############################################

# qui rimuovo gli accapi, spazi inutili e tiro fuori cose come lunghezza della frase, numero di parole etc.

def clean_text(text):
    """toglie spazi inutili e accapi"""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    # Replace escaped newlines
    text = text.replace('\\n', ' ')
    return text

def extract_basic_features(text):
    """pulisce il testo e ricava alcune features di base"""
    cleaned_text = clean_text(text)
    
    char_count = len(cleaned_text)
    word_count = len(cleaned_text.split())
    sentence_count = len(sent_tokenize(cleaned_text))
    

    words = cleaned_text.split()
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # restituisco il dizionario con tutta la roba che ho calcolato
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }

# pulisco
df['review_text'] = df['review_text'].apply(clean_text)

# tiro fuori le features di base e concateno al df così tieni tutto insieme
basic_features = df['review_text'].apply(extract_basic_features)
basic_features_df = pd.DataFrame(basic_features.tolist())
df = pd.concat([df, basic_features_df], axis=1)

print("DOPO PREPROCESSING")
print(df.head(1))

##############################################
################ LUNGHEZZA ###################
##############################################

def calculate_length_score(row):
    """Sputa fuori un punteggio in base alla lunghezza della review"""
    word_count = row['word_count']
    sentence_count = row['sentence_count']
    avg_sentence_length = row['avg_sentence_length']
    
    # una review se è troppo corta male, se troppo lunga male anche quello
    #i punteggi e le soglie le ho tirate fuori dal cappello, poi le modifichiamo se serve
    if word_count < 10:
        word_score = 0.1
    elif word_count <= 30:
        word_score = 0.6
    elif word_count <= 100:
        word_score = 1.0
    elif word_count <= 200:
        word_score = 0.9
    else:
        word_score = 0.7
    
    # stesso di sopra ma con le frasi
    if sentence_count <= 1:
        sentence_score = 0.3
    elif sentence_count <= 3:
        sentence_score = 0.7
    elif sentence_count <= 6:
        sentence_score = 1.0
    else:
        sentence_score = 0.8
    
    # stesso ma con la lunghezza media della frase
    if avg_sentence_length < 5:
        sentence_length_score = 0.4
    elif avg_sentence_length <= 15:
        sentence_length_score = 1.0
    elif avg_sentence_length <= 25:
        sentence_length_score= 0.8
    else:
        sentence_length_score = 0.6
    
    # le sommo insieme dando più importanza al numero di parole
    length_score = (word_score * 0.4 + sentence_score * 0.3 + sentence_length_score * 0.3)
    return length_score


df['length_score'] = df.apply(calculate_length_score, axis=1)

# binario, se il punteggio è maggiore di 0.6 allora è informativa, altrimenti no
df['length_informative'] = df['length_score'] >= 0.6    # arbitrario poi vediamo

print("Length Analysis Results:")
print(f"Classificate come informative (lunghezza): {df['length_informative'].sum()}/{len(df)} ({df['length_informative'].mean():.1%})")

print(df.head(1))


#################################################
################ SPECIFICITA' ###################
#################################################

# questo è quello che pensio sia più importante
# coincidentemente è anche quello che ho fatto più a culo

def calculate_specificity_score(text):
    """sputa fuori un punteggio in base a quanto è \"specifica\" la recensioe"""
    text_lower = text.lower()
    
    # allora in pratica vedo quante volte ci sono parole che indicano aspetti specifici
    # naturalmente le tengo tutte hard-coded e scelte completamente in maniera arbitraria
    
    # Define the base aspect words
    service_words = [
        'service', 'staff', 'employee', 'manager', 'customer service', 'support', 'representative', 'clerk', 'cashier'
    ]
    product_words = [
        'product', 'item', 'selection', 'quality', 'variety', 'availability', 'stock', 'goods', 'merchandise'
    ]
    experience_words = [
        'experience', 'visit', 'appointment', 'session', 'event', 'activity', 'class', 'tour', 'trip'
    ]
    environment_words = [
        'environment', 'atmosphere', 'ambiance', 'clean', 'comfortable', 'space', 'location', 'facility', 'store', 'office', 'building'
    ]
    price_words = [
        'price', 'cost', 'expensive', 'cheap', 'value', 'money', 'dollar', '$', 'budget', 'fee', 'rate', 'charge'
    ]
    time_words = [
        'wait', 'time', 'quick', 'slow', 'minutes', 'hours', 'fast', 'prompt', 'delay', 'schedule', 'appointment'
    ]

    # Helper to get synonyms from WordNet
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word:
                    synonyms.add(synonym)
        return synonyms

    # Build expanded sets with synonyms (flattened)
    def expand_with_synonyms(word_list):
        expanded = set(word_list)
        for word in word_list:
            expanded.update(get_synonyms(word))
        return expanded

    # Expand all aspect word lists
    service_words = list(expand_with_synonyms(service_words))
    product_words = list(expand_with_synonyms(product_words))
    experience_words = list(expand_with_synonyms(experience_words))
    environment_words = list(expand_with_synonyms(environment_words))
    price_words = list(expand_with_synonyms(price_words))
    time_words = list(expand_with_synonyms(time_words))
    
    all_aspect_words = (
        service_words + product_words + experience_words +
        environment_words + price_words + time_words
    )
    
    # qui vedo quante volte ci sono queste parole nel testo
    aspect_mentions = sum(1 for word in all_aspect_words if word in text_lower)

    # qui invece lo divido per 10, se esce più di 10 prende 1
    # naturalmente, il 10 è preso a cazzo
    aspect_score = min(aspect_mentions / 10, 1.0)
    

    #stesso di sopra ma con altre parole
    # li tengo separati perchè sopra sono le parole relative a servizi, prezzi etc
    # mentre qui sono termini che descrivono bene o male come è andata
    descriptive_words = [
        'excellent', 'terrible', 'amazing', 'awful', 'outstanding', 'horrible', 'fantastic', 'disappointing',
        'perfect', 'worst', 'best', 'incredible', 'friendly', 'rude', 'helpful', 'unprofessional', 'attentive',
        'slow', 'quick', 'clean', 'dirty', 'comfortable', 'uncomfortable', 'professional', 'efficient', 'knowledgeable'
    ]
    
    descriptive_words = list(expand_with_synonyms(descriptive_words))
    
    descriptive_count = sum(1 for word in descriptive_words if word in text_lower)
    descriptive_score = min(descriptive_count / 10, 1.0)
    
    # dettaglietti specifici
    # controlloo parole generiche, sperando di azzeccarci per la maggior parte delle volte
    numbers = len(re.findall(r'\b\d+\b', text))  # tiro fuori i numeri
    specific_places = len(re.findall(
        r'\b(table|booth|bar|counter|window|parking|bathroom|desk|room|section|area|floor|entrance|exit|checkout|register)\b',
        text_lower))
    specific_items = len(re.findall(
        r'\b(burger|pizza|salad|coffee|product|item|service|package|deal|offer|appointment|class|session|ticket|order)\b',
        text_lower))
    
    details_score = min((numbers + specific_places + specific_items) / 5, 1.0)
    
    # termini comparativi
    comparison_words = ['better', 'worse', 'compared', 'than', 'like', 'similar', 'different', 'previous']
    comparison_words = list(expand_with_synonyms(comparison_words))
    comparison_count = sum(1 for word in comparison_words if word in text_lower)
    # qui divido per 3 perchè sono meno parole
    comparison_score = min(comparison_count / 3, 1.0)
    
    # li metto tutti insieme
    # do meno importanza ai dettaglietti perchè è quello che mi fido di meno
    specificity_score = (aspect_score * 0.4 + descriptive_score * 0.3 + 
                        details_score * 0.2 + comparison_score * 0.1)
    
    return {
        'aspect_score': aspect_score,
        'descriptive_score': descriptive_score,
        'details_score': details_score,
        'comparison_score': comparison_score,
        'specificity_score': specificity_score
    }

specificity_features = df['review_text'].apply(calculate_specificity_score)
specificity_df = pd.DataFrame(specificity_features.tolist())

df = pd.concat([df, specificity_df], axis=1)

# binario, come sopra
df['specificity_informative'] = df['specificity_score'] >= 0.6

print("Specificity Analysis Results:")
print(f"Classificate come informative (specificità): {df['specificity_informative'].sum()}/{len(df)} ({df['specificity_informative'].mean():.1%})")
print(df.head(1))

##############################################
############### QUALITA' #####################
##############################################

# per qualità intendo facilità di lettura, bilanciamento tra parole emotive e fattuali, struttura grammaticale bla bla bla

def calculate_quality_score(text):
    """Calculate informativeness score based on content quality"""
    
    # qui controllo quant'è facile da leggere la review
    try:
        reading_ease = flesch_reading_ease(text)
        # Optimal readability range: 60-70 (standard)
        if reading_ease < 30:
            readability_score = 0.3
        elif reading_ease < 60:
            readability_score = 0.7
        elif reading_ease <= 70:
            readability_score = 1.0
        elif reading_ease <= 90:
            readability_score = 0.8
        else:
            readability_score = 0.5
    except:
        readability_score = 0.5  # a volte fallisce
    
    text_lower = text.lower()
    
    # Emotional vs factual content
    # Define base emotional and factual words
    emotional_words = ['love', 'hate', 'amazing', 'terrible', 'awesome', 'awful', 'fantastic', 
                      'horrible', 'wonderful', 'disgusting', 'perfect', 'worst', 'best']
    factual_words = ['because', 'since', 'due to', 'reason', 'caused', 'result', 'therefore',
                    'however', 'although', 'despite', 'while', 'whereas', 'specifically']

    # sinonimi da WordNet
    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word:
                    synonyms.add(synonym)
        return synonyms

    #wrapper di sopra per espandere le liste
    def expand_with_synonyms(word_list):
        expanded = set(word_list)
        for word in word_list:
            expanded.update(get_synonyms(word))
        return expanded

    emotional_words = list(expand_with_synonyms(emotional_words))
    factual_words = list(expand_with_synonyms(factual_words))
    
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    factual_count = sum(1 for word in factual_words if word in text_lower)
    
    total_indicators = emotional_count + factual_count

    # se è zero gli infilo 0.5
    if total_indicators == 0:
        balance_score = 0.5
    else:
        # ratio tra fattuali e emotivi
        factual_ratio = factual_count / total_indicators
        if factual_ratio >= 0.4:  # naturalmente più è alto meglio è
            balance_score = 1.0
        elif factual_ratio >= 0.2:
            balance_score = 0.8
        else:
            balance_score = 0.6
    
    # Grammar and structure indicators
    question_marks = text.count('?')
    exclamation_marks = text.count('!')
    punctuation_variety = len(set([c for c in text if c in '.,!?;:']))
    
    # Moderate punctuation use indicates better structure
    if exclamation_marks > 3:
        structure_score = 0.6  # Too many exclamations
    elif punctuation_variety >= 3:
        structure_score = 1.0  # Good variety
    elif punctuation_variety >= 2:
        structure_score = 0.8  # Decent
    else:
        structure_score = 0.5  # Limited structure
    
    # Actionable insights indicators
    actionable_words = ['recommend', 'suggest', 'avoid', 'try', 'should', 'would', 'will',
                       'go back', 'return', 'visit', 'skip', 'worth', 'not worth']

    # Expand actionable_words with synonyms
    actionable_words = list(expand_with_synonyms(actionable_words))
    actionable_count = sum(1 for word in actionable_words if word in text_lower)
    actionable_score = min(actionable_count / 2, 1.0)
    
    # sbatto tutto insieme
    quality_score = (readability_score * 0.3 + balance_score * 0.3 + 
                    structure_score * 0.2 + actionable_score * 0.2)
    
    return {
        'readability_score': readability_score,
        'balance_score': balance_score,
        'structure_score': structure_score,
        'actionable_score': actionable_score,
        'quality_score': quality_score
    }

# Calculate quality scores
quality_features = df['review_text'].apply(calculate_quality_score)
quality_df = pd.DataFrame(quality_features.tolist())

# Add to main dataframe
df = pd.concat([df, quality_df], axis=1)

# Classify reviews based on quality score
df['quality_informative'] = df['quality_score'] >= 0.6

print("Content Quality Analysis Results:")
print(f"Reviews classified as informative (quality): {df['quality_informative'].sum()}/{len(df)} ({df['quality_informative'].mean():.1%})")
print(df.head(1))


##################################################
############## VALUTAZIONE FINALE ################
##################################################

df['combined_score'] = (df['length_score'] * 0.3 + 
                       df['specificity_score'] * 0.4 + 
                       df['quality_score'] * 0.3)

# Final classification
df['is_informative'] = df['combined_score'] >= 0.6

print("Combined Analysis Results:")
print(f"Reviews classified as informative (combined): {df['is_informative'].sum()}/{len(df)} ({df['is_informative'].mean():.1%})")
print(f"Average combined score: {df['combined_score'].mean():.3f}")

# Compare all methods
method_comparison = pd.DataFrame({
    'Length Method': [df['length_informative'].mean()],
    'Specificity Method': [df['specificity_informative'].mean()],
    'Quality Method': [df['quality_informative'].mean()],
    'Combined Method': [df['is_informative'].mean()]
})

print(f"\ntutti e tre:")
for method, percentage in method_comparison.iloc[0].items():
    print(f"{method}: {percentage:.1%}")

# Distribuzione dello score combinato
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(df['combined_score'], bins=30, alpha=0.7, edgecolor='black')
plt.axvline(x=0.6, color='red', linestyle='--', label='Threshold')
plt.xlabel('Combined Informativeness Score')
plt.ylabel('Frequency')
plt.title('Distribution of Combined Scores')
plt.legend()

plt.tight_layout()
plt.show()

print("=== 3 migliori ===\n")
most_informative = df.nlargest(3, 'combined_score')

for i, (idx, row) in enumerate(most_informative.iterrows(), 1):
    print(f"Review {i} (Score: {row['combined_score']:.3f}):")
    print(f"Length: {row['word_count']} words, {row['sentence_count']} sentences")
    print(f"Scores - Length: {row['length_score']:.2f}, Specificity: {row['specificity_score']:.2f}, Quality: {row['quality_score']:.2f}")
    print(f"Text: {row['review_text'][:300]}{'...' if len(row['review_text']) > 300 else ''}")
    print("-" * 80)

print("\n=== 3 peggiori ===\n")
least_informative = df.nsmallest(3, 'combined_score')

for i, (idx, row) in enumerate(least_informative.iterrows(), 1):
    print(f"Review {i} (Score: {row['combined_score']:.3f}):")
    print(f"Length: {row['word_count']} words, {row['sentence_count']} sentences")
    print(f"Scores - Length: {row['length_score']:.2f}, Specificity: {row['specificity_score']:.2f}, Quality: {row['quality_score']:.2f}")
    print(f"Text: {row['review_text'][:300]}{'...' if len(row['review_text']) > 300 else ''}")
    print("-" * 80)


save_df = df[['review_text', 'is_informative', 'combined_score']]

# Salvo sulla repo

fileName = "train_informativeTOT.csv"
os.makedirs("temp", exist_ok=True)
save_df.to_csv(f"temp/{fileName}", index=False)

api = HfApi()
api.upload_file(
    path_or_fileobj=f"temp/{fileName}",
    path_in_repo=f"dataset/{fileName}",
    repo_id=repo,
    repo_type="model"
)
print(f"Saved and uploaded {fileName} to the repo.")

if os.path.exists("temp"):
    shutil.rmtree("temp")