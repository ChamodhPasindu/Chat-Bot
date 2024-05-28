import json
import pickle
import random
import nltk
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Debugging statements
print("Sample bag length:", len(training[0][0]))
print("Sample output_row length:", len(training[0][1]))

# Ensure all bags and output_rows have the same length
for i, (bag, output_row) in enumerate(training):
    if len(bag) != len(words) or len(output_row) != len(classes):
        print(f"Error at index {i}: Bag length {len(bag)}, Expected {len(words)}; Output length {len(output_row)}, Expected {len(classes)}")

random.shuffle(training)
train_x = np.array([t[0] for t in training])
train_y = np.array([t[1] for t in training])

# Now `train_x` and `train_y` should be homogeneous and can be converted to numpy arrays
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')
