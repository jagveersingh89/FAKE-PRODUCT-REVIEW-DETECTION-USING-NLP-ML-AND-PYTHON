import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from preprocess import clean_text

# Load dataset

df = pd.read_csv(
    'dataset/reviews.csv'
)

# Clean review text

df['clean_review'] = df[
    'review'
].apply(clean_text)

# Encode category

category_encoder = LabelEncoder()

df['category_encoded'] = (
    category_encoder.fit_transform(
        df['category']
    )
)

# Encode verified purchase

df['verified_purchase'] = (
    df['verified_purchase']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['verified_purchase'] = (
    df['verified_purchase'].map({

        'yes': 1,
        'no': 0

    })
)

# Encode sentiment

df['sentiment'] = (
    df['sentiment']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['sentiment'] = (
    df['sentiment'].map({

        'positive': 1,
        'neutral': 0,
        'negative': -1

    })
)

# Encode labels

df['label'] = (
    df['label']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['label'] = (
    df['label'].map({

        'real': 0,
        'fake': 1

    })
)

# Numeric conversions

df['rating'] = pd.to_numeric(
    df['rating'],
    errors='coerce'
)

df['account_age_days'] = pd.to_numeric(
    df['account_age_days'],
    errors='coerce'
)

df['helpful_votes'] = pd.to_numeric(
    df['helpful_votes'],
    errors='coerce'
)

# Remove null rows

df = df.dropna()

# TF-IDF

vectorizer = TfidfVectorizer()

text_features = (
    vectorizer.fit_transform(
        df['clean_review']
    )
)

# Numerical features

numerical_features = df[[

    'rating',
    'account_age_days',
    'verified_purchase',
    'helpful_votes',
    'sentiment',
    'category_encoded'

]].values

# Sparse matrix

numerical_features = csr_matrix(
    numerical_features
)

# Combine features

X = hstack([

    text_features,
    numerical_features

])

# Labels

y = df['label']

# Train test split

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42

    )
)

# Random Forest model

model = RandomForestClassifier(

    n_estimators=150,

    random_state=42

)

# Train

model.fit(

    X_train,
    y_train

)

# Predict

predictions = model.predict(
    X_test
)

# Accuracy

accuracy = accuracy_score(

    y_test,
    predictions

)

print(
    f'Accuracy: {accuracy * 100:.2f}%'
)

# Create model folder

os.makedirs(
    'model',
    exist_ok=True
)

# Save files

joblib.dump(

    model,
    'model/fake_review_model.pkl'

)

joblib.dump(

    vectorizer,
    'model/tfidf_vectorizer.pkl'

)

joblib.dump(

    category_encoder,
    'model/category_encoder.pkl'

)

print(
    'Advanced model saved successfully!'
)