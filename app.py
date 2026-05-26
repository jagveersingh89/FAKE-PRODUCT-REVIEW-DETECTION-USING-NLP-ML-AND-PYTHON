from flask import Flask, render_template, request

import pandas as pd
import joblib

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

from preprocess import clean_text

from textblob import TextBlob

app = Flask(__name__)

# =========================
# LOAD TRAINED FILES
# =========================

model = joblib.load(
    'model/fake_review_model.pkl'
)

vectorizer = joblib.load(
    'model/tfidf_vectorizer.pkl'
)

category_encoder = joblib.load(
    'model/category_encoder.pkl'
)

# =========================
# SENTIMENT ANALYSIS
# =========================

def analyze_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return 1

    elif polarity < 0:
        return -1

    else:
        return 0

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():

    return render_template(
        'index.html'
    )

# =========================
# PREDICTION
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    # =========================
    # FORM DATA
    # =========================

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

    # =========================
    # CLEAN TEXT
    # =========================

    cleaned_review = clean_text(
        review
    )

    # =========================
    # TFIDF FEATURES
    # =========================

    text_features = vectorizer.transform(
        [cleaned_review]
    )

    # =========================
    # CATEGORY ENCODING
    # =========================

    try:

        category_encoded = (
            category_encoder.transform(
                [category]
            )[0]
        )

    except:

        category_encoded = 0

    # =========================
    # VERIFIED PURCHASE
    # =========================

    verified_purchase = (

        1 if verified_purchase == 'Yes'

        else 0

    )

    # =========================
    # SENTIMENT
    # =========================

    sentiment = analyze_sentiment(
        review
    )

    # =========================
    # NUMERICAL FEATURES
    # =========================

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

    # =========================
    # SPARSE MATRIX
    # =========================

    numerical_features = csr_matrix(
        numerical_data.values
    )

    # =========================
    # COMBINE FEATURES
    # =========================

    final_features = hstack([

        text_features,

        numerical_features

    ])

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(
        final_features
    )[0]

    # =========================
    # RESULTS
    # =========================

    if prediction == 1:

        result = (
            'Fake Review Detected'
        )

        fake_score = 92

        suspicious_score = 74

        genuine_score = 8

        recommendation = (

            'This review appears highly suspicious '
            'and may contain spam or manipulated content.'

        )

    else:

        result = (
            'Genuine Review'
        )

        fake_score = 12

        suspicious_score = 18

        genuine_score = 91

        recommendation = (

            'This review appears authentic '
            'and trustworthy.'

        )

    # =========================
    # RENDER RESULT PAGE
    # =========================

    return render_template(

        'result.html',

        result=result,

        fake_score=fake_score,

        suspicious_score=suspicious_score,

        genuine_score=genuine_score,

        recommendation=recommendation,

        review=review,

        category=category,

        rating=rating,

        account_age=account_age,

        helpful_votes=helpful_votes

    )

# =========================
# RUN APP
# =========================

if __name__ == '__main__':

    app.run(debug=True)