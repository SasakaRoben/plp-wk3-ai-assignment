# Task3_NLP_spaCy.py (or use in Jupyter cells)
"""
NLP with spaCy:
- Perform Named Entity Recognition (NER)
- Extract product names and brands
- Apply simple rule-based sentiment analysis
"""

# -------------------------
# 1) Imports
# -------------------------
import spacy
from textblob import TextBlob

# Load the small English model (includes NER and POS tagging)
nlp = spacy.load("en_core_web_sm")

# -------------------------
# 2) Sample Amazon product reviews
# -------------------------
reviews = [
    "I absolutely love my new Samsung Galaxy S23! The camera quality is amazing.",
    "The Apple AirPods Pro have terrible battery life. Not worth the price.",
    "I bought a Dell XPS 13 and it performs flawlessly for my daily tasks.",
    "The Sony WH-1000XM5 headphones are super comfortable and the noise cancellation is top-notch!",
    "Avoid the cheap charging cables from BrandX. They stopped working after a week."
]

# -------------------------
# 3) Perform NER (Named Entity Recognition)
# -------------------------
print("=== Named Entity Recognition (NER) ===\n")

for i, review in enumerate(reviews, start=1):
    doc = nlp(review)
    print(f"Review {i}: {review}")
    
    # Extract product-related entities (ORG and PRODUCT are most relevant)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    if entities:
        for text, label in entities:
            print(f"  - Entity: {text:<25} | Label: {label}")
    else:
        print("  - No product or brand entities found.")
    print()

# -------------------------
# 4) Rule-based Sentiment Analysis
# -------------------------
# Using TextBlob for quick polarity scoring
# polarity > 0 -> Positive
# polarity < 0 -> Negative
# polarity = 0 -> Neutral

print("=== Sentiment Analysis ===\n")
for i, review in enumerate(reviews, start=1):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    sentiment = (
        "Positive" if polarity > 0.1 else
        "Negative" if polarity < -0.1 else
        "Neutral"
    )
    print(f"Review {i}: {review}")
    print(f"  Sentiment: {sentiment} (Polarity: {polarity:.2f})\n")

# -------------------------
# 5) Example combined output
# -------------------------
print("=== Summary ===\n")
for i, review in enumerate(reviews, start=1):
    doc = nlp(review)
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    sentiment = (
        "Positive" if polarity > 0.1 else
        "Negative" if polarity < -0.1 else
        "Neutral"
    )
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]
    
    print(f"Review {i}:")
    print(f"  Text: {review}")
    print(f"  Entities: {entities if entities else 'None'}")
    print(f"  Sentiment: {sentiment} (Polarity={polarity:.2f})")
    print("-" * 80)
