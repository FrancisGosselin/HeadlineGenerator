
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
parser.add_argument('--manual', action='store_true', dest='manual', help='program will prompt for input to start a headline')

args = parser.parse_args()

def progress_bar(percentage, length):
    load_progress = '#'*int(percentage/100*length) + '-'*(length - int(percentage/100*length))
    return "\r[%s]  %i%% trained" % (load_progress, percentage)

class Model():

    vocabulary = set()
    occurence_table = dict({})
    tokenizer = r'''(?x)(?:[A-Z]\.)+|\w+(?:-\w+)*|\$?\d+(?:\.\d+)?%?|\.\.\.|[][.,;"'?():_`-]'''

    def __init__(self, ngram):
        self.ngram = ngram

    def load(self, filename):
        with open(filename + '.pkl', 'rb') as f:
            data =  pickle.load(f)
            self.occurence_table = data["occurence_table"]
            self.vocabulary = data["vocabulary"]
            self.ngram = data["ngram"]

    def save(self, file):
        
        obj = {"vocabulary": dict.fromkeys(self.vocabulary), "occurence_table": self.occurence_table, "ngram": self.ngram }

        with open( file + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def train(self, filename, maxrows):


        for chunk in pd.read_csv(filename, chunksize=maxrows):

            for index, row in chunk.iterrows():
                if (index %10000 == 0):
                    sys.stdout.write(progress_bar(index/chunk.size*200, 30))
                    sys.stdout.flush()

                self.__processline__(row['headline_text'])

                

            sys.stdout.write(progress_bar(100, 30))
            sys.stdout.flush()
            break
        print("")
        print("Training Done!")

    def __processline__(self, line) :

        tokens = re.findall(self.tokenizer, line)
        tokens = ['BOS'] + tokens + ['EOS']

        for i, word in enumerate(tokens):
            if word not in self.vocabulary :
                self.vocabulary.add(word)
            if i > 0:
                for cntxt_size in range(1, min(i, self.ngram - 1) + 1):
                    if tuple(tokens[i-cntxt_size:i]) not in self.occurence_table:
                        self.occurence_table[tuple(tokens[i-cntxt_size:i])] = dict({})
                    if word not in self.occurence_table[tuple(tokens[i-cntxt_size:i])]:
                        self.occurence_table[tuple(tokens[i-cntxt_size:i])][word] = 0
                    self.occurence_table[tuple(tokens[i-cntxt_size:i])][word] += 1

    def generate(self, start = ''):
        phrase = re.findall(self.tokenizer, start)
        phrase = ['BOS'] + phrase

        for i in range(len(phrase), 20):
            cntxt_table = None
            for cntxt_size in reversed(range(1, min(i, self.ngram - 1) + 1)):
                if tuple(phrase[i-cntxt_size:i]) in self.occurence_table:
                    cntxt_table = self.occurence_table[tuple(phrase[i-cntxt_size:i])]
            
            if not cntxt_table:
                phrase.append(random.choice(list(self.vocabulary())))
            
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
    print('Saving model...')
    model.save(args.filesave)

if args.fileload:
    print('Loading model...')
    model = Model(3)
    model.load(args.fileload)

if args.headlines:
    for i in range(int(args.headlines)):
        print(model.generate())

if args.manual:
    while True:
        print(model.generate(input('Enter a begging of a headline: ').lower()))
    