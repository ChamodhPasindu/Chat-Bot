import random
import json
import pickle
import numpy as np
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


# Load data and model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# List of greeting messages
greeting_messages = [
    "Hello! How can I assist you today?",
    "Hi there! Feel free to ask any questions about our college.",
    "Greetings! How can I help you today?",
    "Welcome! What would you like to know about our college?",
    "Hey! Ask me anything about our college."
]

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    global result
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/api/data', methods=['GET'])
def greet_bot():
    random_greeting = random.choice(greeting_messages)
    return create_response(random_greeting)

@app.route('/api/data', methods=['POST'])
def start_bot():
        data = request.get_json()
        message = data.get('message')
        action = data.get('action') 
        print(message,action)

        ints = predict_class(message)
        if ints:
            res = get_response(ints, intents)
            return create_response(res)

        else:
            return create_response("Sorry, I didn't understand that. Can you please rephrase?")

           
def create_response(reply):
    base_request = {
        "reply": reply,
        "action": "bot",
    }
    return jsonify(base_request)


if __name__ == '__main__':
    app.run(debug=True)