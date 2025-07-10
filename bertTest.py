from transformers import BertTokenizer, BertForSequenceClassification

model_dir = "lPen88/webProject"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)


def predict_informativeness(text):
    """
    Predicts the informativeness of a given text using a pre-trained BERT model.
    
    Args:
        text (str): The input text to evaluate.
        
    Returns:
        int: The predicted informativeness score (0 or 1).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=1).item()
    return predicted_class


texts = [
    "This place stinks!",
    "I have to say, not too bad",
    "I love this place, the food is amazing and the service is great!",
    "I was immediately displeased by the snaggle toothed hostess and her surly demeanor. The air conditioning was non existent, which is a hangover nightmare. While waiting for a table the bartender ignored us for ten minutes and couldn't even transfer our drinks to the table. He also casually instructed me that one must pay per beer - duh! I'm not a beer drinking novice. The acne-ridden server was friendly enough, but seemed forlorn and was overtly sweating. Bringing drinks to a table isn't neuroscience and I don't drink tap water. After sitting at our table for a few minutes it was very apparent that the entire serving staff is haggard. Appetizer - crab artichoke dip. Bland. Where's the crab? Also, did a toddler arrange the plate? Apparently. Meal - fish tacos. Came quickly and surprisingly flavorful. However, our server never once checked on our status and we had to beckon him to fulfill our needs. I shouldn't have to ask for napkins at the point of need. No tip, never coming back."
    "The food was mediocre at best, and the service was slow. I expected more from this restaurant.",
    "The ambiance was lovely, and the staff were attentive. The pasta was cooked to perfection, and the dessert was a delightful end to the meal. All very cheap to boot!",
    "I had a positive experience with this shop. I needed something framed at the last minute, and within extremely precise parameters (that weren't easy to abide by), and they not only got the work done correctly and on time, but at a reasonable price. \n\nThe staff is a little gruff, but not at all unfriendly, and I am happy to have found a local shop that I will return to."
]

for text in texts:
    score = predict_informativeness(text)
    print(f"Text: {text}\nPredicted Informativeness Score: {score}\n")