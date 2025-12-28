import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)         # punctuation
    text = re.sub(r"\s+", " ", text).strip()     # collapse spaces
    return text

