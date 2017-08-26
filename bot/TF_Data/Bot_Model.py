# Imports for Natural Language Processing(NLP)
#--------------------------------------------------------------------
# nltk - Natural Language Tool Kit
# LancasterStemmer is  a popular stemming algorithm
# Stemming is the process of finding the root word ,
# given a series of words(contextually)
# Wikipedia has an example -
# A stemming algorithm should identify the string "cats" 
# (and possibly "catlike", "catty" etc.) as based on the root "cat"
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Imports for TF
import numpy as np 
import tflearn
import tensorflow as tf 
import random

# Importing the language file for the bot
import json
with open('language.json') as json_data:
	language = json.load(json_data)

# Orgainzing words, documents and 
words = []
classes = []
documents = []
ignore_words = ['?']

# looping through each item in our language
for item in language['language']:
	for pattern in item['patterns']:
		# Tokenize each word
		w = nltk.word_tokenize(pattern)
		
		# add this to our word list
		words.extend(w)

		# populating the documents with corresponding tag
		documents.append((w, item['tag']))

		#populating the class list
		if item['tag'] not in classes:
			classes.append(item['tag'])

# ---------TEST---------------
print("\nWORDS:")
print("-----------")
print(words)
print("\nDOCUMENTS:")
print("-----------")
print(documents)
print("\nCLASSES:")
print("-----------")
print(classes)

# Stem and lower each word, then remove duplicates
words = [stemmer.stem(w.lower()) for w in words 
									if w not in ignore_words]
words = sorted(list(set(words)))

# ---------TEST---------------
print("\nWORDS AFTER STEMMING:")
print("-----------")
print(words)

# Remove duplicates
classes = sorted(list(set(classes)))

# ---------TEST---------------
print("\nCLASSES AFTER SORTING:")
print("-----------")
print(classes)

print(len(documents), " documents")

# Now, from document of words we need to transform to 
# tensors of numbers
# -----------------------------------------------------

# Creating our trainig data
training = []
output = []

# create an empty array for the output
output_empty = [0] * len(classes)

# populating  the training set, bag of words for each sentence
for doc in documents:
	bag = []

	# list of tokenized words for the pattern
	pattern_words = doc[0]

	# Start stemming each word
	pattern_words = [stemmer.stem(word.lower()) 
					  for word in pattern_words]

	# Create a bag of words array
	# Note: bag-of-words is a simplifying representation used in 
	# NLP. It is simply , breaking down a sentence into list 
	# of words.
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	# output is a '0' for each tag and '1' for current tag
	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1

	training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# ---------TEST---------------
# the tarinaing data is now in x,y form where x and y both are sets
print("TRAINING")
print("\n", training)

# create train and test lists
# here only the x part of training is taken
train_x = list(training[:, 0])
# here only the y part of training is taken 
train_y = list(training[:, 1])

# ---------TEST---------------
print("TRAINING -x")
print("\n",train_x)
print("TRAINING -y")
print("\n",train_y)
print(words)

# Lets build the model
#--------------------------------------------
tf.reset_default_graph()

# build the neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]),
							  activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')

# start training , apply gradient descent algorithm
model.fit(train_x, train_y, n_epoch=1000, batch_size=8,
		   show_metric=True)
model.save('model.tflearn')

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

p = bow("Is anyone there?", words)
print(p)
print(classes)

print(model.predict([p]))


# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# After this you might see a training_data file in
# the repository