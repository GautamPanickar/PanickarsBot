# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# Restore all our data structures from Bot_Model.py
import pickle
data = pickle.load(open("C:/Users/User/Documents/Django_Workspace/PanickarsBot/bot/TF_Data/training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importing the langugage for my bot
import json
with open("C:/Users/User/Documents/Django_Workspace/PanickarsBot/bot/TF_Data/language.json") as json_data:
        language = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')



# we'll use the same methods as in model

#The method for cleaning up the sentence
def clean_up_sentence(sentence):
        # tokenize the pattern
        senetence_words = nltk.word_tokenize(sentence)

        #Atart stemming each word
        senetence_words = [stemmer.stem(word.lower()) for word in senetence_words]
        return senetence_words

# return bag of words array: 0 or 1 for each word in the bag
# that exists in the sentence
def bow(sentence, words, show_details=False):
        #tokenize the pattern
        senetence_words = clean_up_sentence(sentence)

        #bow
        bag = [0]*len(words)
        for s in senetence_words:
                for i,w in enumerate(words):
                        if w == s:
                                bag[i] = 1
                                if show_details:
                                        print("found in bag: %s" % w)

        return (np.array(bag))

# loading the saved tf model
model.load('C:/Users/User/Documents/Django_Workspace/PanickarsBot/bot/TF_Data/model.tflearn')

#---------------------------------------------
# The response processor
#---------------------------------------------
# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25

def classify(sentence):
        # generate probabilities for the model
        results = model.predict([bow(sentence, words)])[0]

        # filter out predictions below a ERROR_THRESHOLD
        results = [[i,r] for i,r in enumerate(results) 
                                if r > ERROR_THRESHOLD ]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for r in results:
                return_list.append((classes[r[0]], r[1]))

        #return tuple of language and the probability
        return return_list

def response(sentence, userID='sonnynefarious', show_details=False):
        results = classify(sentence)

        # If there is a classification, then find the matching 
        # language
        if results:
                #loop as  long as there are matches to process
                while results:
                        for i in language['language']:
                                #find a tag matchong the first result
                                if i['tag'] == results[0][0]:
                                        # set context for this langugae if necessary
                                        if 'context_set' in i:
                                                if show_details:
                                                        print ('context:', i['context_set'])
                                                context[userID] = i['context_set']
                                                        
                                        # check if this intent is contextual and applies to this user's conversation
                                        if not 'context_filter' in i or (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                                                if show_details:
                                                        print ('tag:', i['tag'])
                                                        
                                                # a random response from the languagr
                                                return random.choice(i['responses'])
                        results.pop(0)
