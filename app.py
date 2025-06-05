
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Sample training data
data = pd.DataFrame({
    'text': [
        'I love this product!',
        'This is terrible.',
        'I am happy with the service.',
        'I am so disappointed.',
        'Absolutely fantastic!',
        'Not good at all.',
        'It was okay.',
        'Horrible experience.',
        'I like it.',
        'Worst purchase ever.'
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'neutral',
        'negative',
        'positive',
        'negative'
    ]
})

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
model = MultinomialNB()
model.fit(X, y)

# App UI
st.title("Sentiment Analysis App")
st.markdown("Enter a message or sentence to classify its sentiment.")

user_input = st.text_area("Type your message here:")

if st.button("Predict Sentiment"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    st.success(f"Predicted Sentiment: {prediction.capitalize()}")
