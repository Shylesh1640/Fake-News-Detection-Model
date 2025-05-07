import joblib
# Load model and vectorizer
model = joblib.load("xgb_fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Sample prediction
sample = ["NASA just exposed the truth about elections."]
vec = vectorizer.transform(sample)
prediction = model.predict(vec)

print("Prediction:", "Real" if prediction[0] == 1 else "Fake")
