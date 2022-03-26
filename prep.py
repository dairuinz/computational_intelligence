import pandas as pd
import re
import numpy as np

def bow():
    vocab_df = pd.read_csv('Data/vocabs.txt', sep=',', header=None)
    vocab_df.columns = ['word', 'id']
    # print(vocab_df.head())

    df = pd.read_csv('Data/train-data.dat', sep=',', header=None)
    df.columns = ['sentence']
    df['sentence'] = df['sentence'].str.replace('<.*?>', '', regex=True)        #deletes <int>
    # print(df.head())
    df = df[0:100]
    # print(df)

    dict = {}                                       #creates dictionary from vocabs file
    with open('Data/vocabs.txt') as file:
        for line in file:
            (key, value) = line.split(', ')
            dict[key] = int(value)

    # print(dict.values())

    X = []                                          #creates bag of words array
    for y in df.sentence:
        x_counter = [0] * 8519
        for w in y.split():
            for i in range(8520):
                if int(w) == i:
                    x_counter[i] = x_counter[i] + 1
        X.append(x_counter)
    # print(X)
    # print(len(X), 'x', len(X[0]), sep='')

    X = pd.DataFrame(X)

    return X

def out():
    y = pd.read_csv('Data/train-label.dat', sep=',', header=None)
    y.columns = ['label']
    y = y[0:100]
    y = y['label'].str.split(' ', expand=True)
    y = y.astype(int)

    return y
