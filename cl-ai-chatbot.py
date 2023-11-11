# pre-trained model for execution instead of undergoing the process of retraining.

import json
import pickle
import nltk
import numpy as np
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import random

stemmer = LancasterStemmer()

# json loader
with open("bot-intents.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as file:
        words, labels, training, output = pickle.load(file)
except:
    words = []  # all words in patterns
    labels = []  # all tags

    docs_wrds = []  # words
    docs_pat = []  # pattern: hi hello seeya

    # data -> intents <multiple dicts>
    # -> choosing pattern as key -> values = input

    # stemming to the root via stemmer
    for intent in data['intents']:
        for pattern in intent['patterns']:
            ws = nltk.word_tokenize(pattern)  # conv-to-list/tokens
            words.extend(ws)  # appending on steroids
            docs_wrds.append(ws)  # token words
            docs_pat.append(intent['tag'])  # use intent tag as the label

            # print(intent['tag'] + ' : ' + pattern)  # run this to get line 31,32

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']  # stemming the words in pattern
    words = sorted(list(set(words)))  # sorted unique elements

    labels = sorted(labels)

    # creating bag of words via onehotecoded <space consuming>

    training = []
    output = []

    out_Empty = [0 for _ in range(len(labels))]  # [0,0,0,0,0,0]

    for x, doc in enumerate(docs_wrds):  # x just gives s.no
        bag = []
        stmd_Tags = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in stmd_Tags:  # if word from that tag is there then encode it
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_Empty[:]
        output_row[labels.index(docs_pat[x])] = 1

        training.append(bag)
        output.append(output_row)

    # converting data into arrays for the model

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as file:
        pickle.dump((words, labels, training, output), file)

# define model
net = tflearn.input_data(shape=[None, len(training[0])])  # input shape length
# all training input -> same size so size of training[0] = all training
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(labels), activation='softmax')  # output layer prob for each output
net = tflearn.regression(net)

model = tflearn.DNN(net)

# fitting the model

try:
    model.load('model_cbot.tflearn')
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model_cbot.tflearn")


# Prediction via input

def b_o_w(s, words_xd):
    bag_xd = [0 for _ in range(len(words_xd))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for aiml in s_words:
        for i, w in enumerate(words):
            if w == aiml:
                bag_xd[i] = 1

    return np.array(bag_xd)


def chatting_uwu():
    global respsssss
    print('SPEAK HUMAN!: OR type <exit> to quit')
    while True:
        inpt = input('You: ')
        if inpt.lower() == "exit":
            break
        results = model.predict([b_o_w(inpt, words)])
        # print(results)  # output to <Hello> in probab since softmax used ->
        # [[1.0677049e-02 1.4385157e-02 9.7308862e-01 1.7872381e-03 2.9812150e-05 3.2123040e-05]]
        results_index = np.argmax(results)
        # print(labels[results_index])  # now giving only that tag <domain>
        for tgt in data['intents']:
            if tgt['tag'] == labels[results_index]:
                respsssss = tgt['responses']
        print(random.choice(respsssss))


chatting_uwu()
