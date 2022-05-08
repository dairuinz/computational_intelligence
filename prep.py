import pandas as pd
import re
import numpy as np

def bow(df):
    vocab_df = pd.read_csv('Data/vocabs.txt', sep=',', header=None)
    vocab_df.columns = ['word', 'id']

    dict = {}                        # creates dictionary from vocabs file
    with open('Data/vocabs.txt') as file:
        for line in file:
            (key, value) = line.split(', ')
            dict[key] = int(value)

    X = []                                #creates bag of words array
    for y in df.sentence:
        x_counter = [0] * 8519
        for w in y.split():
            for i in range(8520):
                if int(w) == i:
                    x_counter[i] = x_counter[i] + 1
        X.append(x_counter)
    X = pd.DataFrame(X)         #puts array to dataframe

    return X

def out():
    y = pd.read_csv('Data/train-label.dat', sep=',', header=None)       #reads output data
    y.columns = ['label']       #names output data column
    y = y[0:100]
    y = y['label'].str.split(' ', expand=True)          #splits data into 20 columns
    y = y.astype(int)       #makes output data integers

    return y
