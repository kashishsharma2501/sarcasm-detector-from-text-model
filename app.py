from flask import Flask, render_template, request
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model and tokenizer from local 'model' folder
model = TFBertForSequenceClassification.from_pretrained('model')
tokenizer = BertTokenizer.from_pretrained('model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    prediction = tf.nn.softmax(logits, axis=1)
    pred_label = np.argmax(prediction, axis=1)[0]

    result = "Sarcastic 😏" if pred_label == 1 else "Not Sarcastic 🙂"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
