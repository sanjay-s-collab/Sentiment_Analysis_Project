import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()

model = load_model("sentiment_model.keras")

review = input("Enter Review: ")

encoded_review = []

for word in review.lower().split():
 if word in word_index:
   encoded_review.append(word_index[word] + 3)

encoded_review = pad_sequences([encoded_review], maxlen=200)

prediction = model.predict(encoded_review)

score = prediction[0][0]

print("\nPrediction Score:", score)

if score > 0.5:
   print("Positive Review 😊")
else:
   print("Negative Review 😞")
