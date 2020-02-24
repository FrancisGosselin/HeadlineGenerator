
import pandas as pd
import re
import sys
import random
import numpy as np
import json
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store', dest='filesave', help='name of the file that will be created')
parser.add_argument('--train', action='store', dest='train', help='name of the training dataset file')
parser.add_argument('--ngram', action='store', dest='ngram', help='The size of the ngram model to train')
parser.add_argument('--load', action='store', dest='fileload', help='loads the model')
parser.add_argument('--print', action='store', dest='headlines', help='print a specific amount of generated headlines')

args = parser.parse_args()

class Model():

    vocabulary = set()
    occurence_table = dict({})

    def __init__(self, ngram):
        self.cntxt_size = ngram - 1
        self.ngram = ngram

    def load(self, filename):
        with open(filename + '.pkl', 'rb') as f:
            data =  pickle.load(f)
            self.occurence_table = data["occurence_table"]
            self.vocabulary = data["vocabulary"]
            self.ngram = data["ngram"]
            self.cntxt_size = self.ngram - 1

    def save(self, file):
        
        obj = {"vocabulary": dict.fromkeys(self.vocabulary), "occurence_table": self.occurence_table, "ngram": self.ngram }

        with open( file + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def train(self, filename, maxrows):

        tokenizer = r'''(?x)(?:[A-Z]\.)+|\w+(?:-\w+)*|\$?\d+(?:\.\d+)?%?|\.\.\.|[][.,;"'?():_`-]'''

        for chunk in pd.read_csv(filename, chunksize=maxrows):

            for index, row in chunk.iterrows():
                if (index %10000 == 0):
                    print(index/chunk.size*200, "% trained")

                tokens = re.findall(tokenizer, row['headline_text'])
                tokens = ['BOS'] + tokens + ['EOS']

                for i, word in enumerate(tokens):
                    if word not in self.vocabulary :
                        self.vocabulary.add(word)
                    if i >= self.cntxt_size :
                        if tuple(tokens[i-self.cntxt_size:i]) not in self.occurence_table:
                            self.occurence_table[tuple(tokens[i-self.cntxt_size:i])] = dict({})
                        if word not in self.occurence_table[tuple(tokens[i-self.cntxt_size:i])]:
                            self.occurence_table[tuple(tokens[i-self.cntxt_size:i])][word] = 0
                        self.occurence_table[tuple(tokens[i-self.cntxt_size:i])][word] += 1

            break
        print("")
    def generate(self):
        while True:
            phrase = list(random.choice(list(self.occurence_table.keys())))
            if phrase[0] == 'BOS':
                break

        for i in range(self.cntxt_size, 20):

            cntxt_table = self.occurence_table[tuple(phrase[i-self.cntxt_size:i])]
            total = sum(cntxt_table.values())
            rand_number = random.randint(0, total - 1)

            counter = 0
            for word, occurence in  cntxt_table.items():
                counter += occurence
                if rand_number <= counter: 
                    phrase.append(word)
                    break
            if phrase[-1] == 'EOS':
                break
        
        return " ".join(phrase).replace('BOS', '').replace('EOS', '')

if args.train:
    if not args.ngram:
        raise ValueError("The value of the ngram must be passed")

    model = Model(int(args.ngram))
    model.train(args.train, 2000000)

if args.filesave:
    model.save(args.filesave)

if args.fileload:
    model = Model(3)
    model.load(args.fileload)

if args.headlines:
    for i in range(int(args.headlines)):
        print(model.generate())
    