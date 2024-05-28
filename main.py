import random
import json
import pickle
import numpy as np
import os
import sys

import tkinter as tk
from tkinter import scrolledtext

import datetime

import nltk
from nltk.data import find
from nltk.stem import WordNetLemmatizer

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')

try:
    find('tokenizers/punkt', paths=[nltk_data_path])
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    find('corpora/wordnet', paths=[nltk_data_path])
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
lemmatizer = WordNetLemmatizer()

# Determine the base path to use for finding files
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

# Load data and model
intents = json.loads(open(os.path.join(base_path, 'intents.json')).read())
words = pickle.load(open(os.path.join(base_path, 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(base_path, 'classes.pkl'), 'rb'))
model = load_model(os.path.join(base_path, 'chatbotmodel.h5'))

# List of greeting messages
greeting_messages = [
    "Hello! How can I assist you today?",
    "Hi there! Feel free to ask any questions about Esoft Metro Campus.",
    "Greetings! How can I help you today?",
    "Welcome! What would you like to know about Esoft Metro Campus?",
    "Hey! Ask me anything about Esoft Metro Campus?"
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

def get_bot_response(message):
        ints = predict_class(message)
        if ints:
            res = get_response(ints, intents)
            return res
        else:
            return "Sorry, I didn't understand that. Can you please rephrase?"

def send_message():
    user_message = user_input.get()
    if user_message.strip() != "":
        insert_message(user_message, "user")
        bot_response = get_bot_response(user_message)
        insert_message(bot_response, "bot")
        chat_log.yview(tk.END)
        user_input.delete(0, tk.END)
        if any(word in user_message.lower() for word in ["bye", "goodbye"]):
            root.after(2000, root.quit)  # Wait for 2 seconds before closing the application

def insert_message(message, sender):
    chat_log.config(state=tk.NORMAL)

    frame = tk.Frame(chat_log, bg="#f0f0f0", bd=0)
    
    # Get current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if sender == "user":
        name_label = tk.Label(frame, text=f"You [{timestamp}]", bg="#f0f0f0", fg="#1f77b4", font=("Arial", 10, "bold"))
        name_label.pack(anchor='e', padx=5, pady=(5, 0))
        message_label = tk.Label(frame, text=message, bg="#dcf8c6", fg="#000000", font=("Arial", 12), padx=10, pady=5, wraplength=250, justify='left')
        message_label.pack(anchor='e', padx=5, pady=(0, 5))
        frame.pack(anchor='e', fill='x', padx=10, pady=5)
        frame.pack(side='right', fill='x', padx=10, pady=5, anchor='e')
    else:
        name_label = tk.Label(frame, text=f"E-Bot [{timestamp}]", bg="#f0f0f0", fg="#ff7f0e", font=("Arial", 10, "bold"))
        name_label.pack(anchor='w', padx=5, pady=(5, 0))
        message_label = tk.Label(frame, text=message, bg="#ffffff", fg="#000000", font=("Arial", 12), padx=10, pady=5, wraplength=250, justify='left')
        message_label.pack(anchor='w', padx=5, pady=(0, 5))
        frame.pack(anchor='w', fill='x', padx=10, pady=5)
        frame.pack(side='left', fill='x', padx=10, pady=5, anchor='w')

    chat_log.window_create(tk.END, window=frame)
    chat_log.insert(tk.END, "\n", sender)
    chat_log.config(state=tk.DISABLED)

def insert_greeting():
    greeting_message = random.choice(greeting_messages)
    insert_message(greeting_message, "bot")

# Create main window
root = tk.Tk()
root.title("Esoft AI Assistant")
root.geometry("500x600")
root.configure(bg="#f0f0f0")  # Set background color

# Create chat log (scrolled text)
chat_log = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD, bg="#f0f0f0", fg="#000000", font=("Arial", 12))
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Insert initial greeting message
insert_greeting()

# Create a frame for user input and send button
input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(padx=10, pady=10, fill=tk.X)

# Create user input field
user_input = tk.Entry(input_frame, width=40, bg="#ffffff", fg="#000000", font=("Arial", 12))
user_input.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
user_input.bind("<Return>", lambda event: send_message())  # Send message on Enter key press

# Create send button
send_button = tk.Button(input_frame, text="Send", command=send_message, bg="#056e2f", fg="#ffffff", font=("Arial", 12))
send_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the application
root.mainloop()
