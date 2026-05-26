import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

# Create stemmer object
stemmer = PorterStemmer()

# Load stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split words
    words = text.split()

    # Store cleaned words
    cleaned_words = []

    for word in words:

        # Remove stopwords
        if word not in stop_words:

            # Apply stemming
            cleaned_words.append(
                stemmer.stem(word)
            )

    # Join words again
    return ' '.join(cleaned_words)