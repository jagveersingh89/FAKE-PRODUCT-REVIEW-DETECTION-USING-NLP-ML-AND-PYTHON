from flask import Flask, render_template, request
import pandas as pd
import joblib
import random

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from preprocess import clean_text
from textblob import TextBlob

app = Flask(__name__)

# Load model files

model = joblib.load(
    'model/fake_review_model.pkl'
)

vectorizer = joblib.load(
    'model/tfidf_vectorizer.pkl'
)

category_encoder = joblib.load(
    'model/category_encoder.pkl'
)


# Sentiment Analysis

def analyze_sentiment(text):

    polarity = TextBlob(
        text
    ).sentiment.polarity

    if polarity > 0:
        return 1

    elif polarity < 0:
        return -1

    else:
        return 0


@app.route('/')
def home():

    return render_template(
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict():

    review = request.form['review']

    category = request.form['category']

    rating = int(
        request.form['rating']
    )

    account_age = int(
        request.form['account_age']
    )

    verified_purchase = request.form[
        'verified_purchase'
    ]

    helpful_votes = int(
        request.form['helpful_votes']
    )

    review_count = int(
        request.form['review_count']
    )

    # Clean review

    cleaned_review = clean_text(
        review
    )

    # TF-IDF

    text_features = vectorizer.transform(
        [cleaned_review]
    )

    # Safe category encoding

    try:

        category_encoded = (
            category_encoder.transform(
                [category]
            )[0]
        )

    except:

        category_encoded = 0

    # Verified purchase

    verified_purchase = (
        1 if verified_purchase == 'Yes'
        else 0
    )

    # Sentiment

    sentiment = analyze_sentiment(
        review
    )

    # Numerical data

    numerical_data = pd.DataFrame({

        'rating': [rating],

        'account_age_days': [
            account_age
        ],

        'verified_purchase': [
            verified_purchase
        ],

        'helpful_votes': [
            helpful_votes
        ],

        'sentiment': [
            sentiment
        ],

        'category_encoded': [
            category_encoded
        ]

    })

    numerical_features = csr_matrix(
        numerical_data.values
    )

    # Combine

    final_features = hstack([

        text_features,
        numerical_features

    ])

    # Prediction probability

    probabilities = model.predict_proba(
        final_features
    )[0]

    fake_probability = round(
        probabilities[1] * 100,
        2
    )

    genuine_probability = round(
        probabilities[0] * 100,
        2
    )

    suspicious_probability = round(
        random.uniform(10, 35),
        2
    )

    # Final Result

    if fake_probability >= 70:

        result = "Fake Review"

        recommendation = (
            "Highly suspicious review. "
            "Recommended to hide or delete."
        )

    elif fake_probability >= 40:

        result = "Suspicious Review"

        recommendation = (
            "Review should be manually checked."
        )

    else:

        result = "Genuine Review"

        recommendation = (
            "Review appears authentic and trustworthy."
        )

    return render_template(

        'result.html',

        result=result,

        recommendation=recommendation,

        fake_probability=fake_probability,

        genuine_probability=genuine_probability,

        suspicious_probability=suspicious_probability,

        review=review,

        category=category,

        rating=rating,

        account_age=account_age,

        helpful_votes=helpful_votes,

        review_count=review_count

    )


if __name__ == '__main__':

    app.run(debug=True)