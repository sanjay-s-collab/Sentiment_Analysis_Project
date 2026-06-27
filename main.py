import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# Load IMDb Dataset

vocab_size = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad Sequences

maxlen = 200

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print("Training Samples:", len(x_train))
print("Testing Samples:", len(x_test))

# Build Model

model = Sequential([
Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
LSTM(64),
Dense(1, activation='sigmoid')
])

# Compile Model

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)

# Train Model

history = model.fit(
x_train,
y_train,
epochs=3,
batch_size=64,
validation_split=0.2
)

# Save Model

model.save("sentiment_model.keras")

# Evaluate Model

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)

# Accuracy Graph

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train', 'Validation'])
plt.savefig("accuracy.png")
plt.show()

# Loss Graph

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'Validation'])
plt.savefig("loss.png")
plt.show()

print("\nFiles Created:")
print("- sentiment_model.keras")
print("- accuracy.png")
print("- loss.png")
