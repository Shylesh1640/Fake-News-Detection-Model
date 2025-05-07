# Optional: Install required packages if not already installed
# pip install xgboost seaborn wordcloud joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the Dataset
df = pd.read_csv("synthetic_fake_news_15000.csv")
print("Label Distribution:\n", df['label'].value_counts())

# Step 2: Visualize Label Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Fake vs Real News Distribution")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.xlabel("News Type")
plt.ylabel("Count")
plt.show()

# Step 3: Generate Word Clouds
fake_words = " ".join(df[df.label == 0]['text'])
real_words = " ".join(df[df.label == 1]['text'])

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(max_words=500, background_color='white').generate(fake_words))
plt.title("Fake News Word Cloud")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(max_words=500, background_color='white').generate(real_words))
plt.title("Real News Word Cloud")
plt.axis('off')
plt.show()

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Step 6: Train XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_vec, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test_vec)

# Step 8: Evaluation
print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Save Model and Vectorizer
joblib.dump(model, "xgb_fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully as 'xgb_fake_news_model.pkl' and 'tfidf_vectorizer.pkl'")

