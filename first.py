import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import pickle

# Load the model
model = pickle.load(open('modelf.pkl', 'rb'))

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define the max sequence length
max_len = 19  # Adjust based on your model

# Initialize Flask app
app = Flask(__name__)



def predict_top_words(model, tokenizer, text, top_k=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    predictions = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_words = [word for word, index in tokenizer.word_index.items() if index in top_indices]
    return top_words

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    print(text)
    top_words = predict_top_words(model, tokenizer, text)
    return render_template('index.html', original_text=text, top_words=top_words)

if __name__ == '__main__':
    app.run(debug=True)