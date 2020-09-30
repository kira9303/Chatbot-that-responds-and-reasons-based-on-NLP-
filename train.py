import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('D:/Chatbot/intents_complete.json').read()
intents = json.loads(data_file)

#print(intents)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #print(words)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#print(words)
#print(classes)
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print("/n")

#print(documents)

print("  /n ")

#print (len(classes), "classes", classes)

print("  /n ")

#print (len(words), "unique lemmatized words", words)

print("  /n ")


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# initializing training data
training = []
output_empty = [0] * len(classes)
#print(output_empty)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    print(pattern_words)
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    #print(output_row)
    output_row[classes.index(doc[1])] = 1
    #print(output_row)
    

    training.append([bag, output_row])
    #print(training)
    #print(bag)
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
#print(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
#print(train_x)
train_y = list(training[:,1])
#print(train_y)
print("Training data created")

print("bag of data is: ")
print(" \n")
print(len(bag))
print("length of words is: ")
print(len(words))

print(" \n")
print("length of input is: ")
print("  {}".format(len(train_x[0])))

#print(len(train_x[0]))




model = Sequential()
model.add(Dense(164, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(128, activation= 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=130, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")


