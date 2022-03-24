import pandas as pd
import re
import numpy as np

def bow():
    vocab_df = pd.read_csv('Data/vocabs.txt', sep=',', header=None)
    vocab_df.columns = ['word', 'id']
    # print(vocab_df.head())

    df = pd.read_csv('Data/test-data.dat', sep=',', header=None)
    df.columns = ['sentence']
    df['sentence'] = df['sentence'].str.replace('<.*?>', '', regex=True)        #deletes <int>
    # print(df.head())
    df = df[0:1000]
    # print(df)

    dict = {}                                       #creates dictionary from vocabs file
    with open('Data/vocabs.txt') as file:
        for line in file:
            (key, value) = line.split(', ')
            dict[key] = int(value)

    # print(dict.values())

    X = []
    for y in df.sentence:
        for w in y.split():
            x_counter = [0] * 8519
            for i in range(8520):
                if int(w) == i:
                    x_counter[i] = x_counter[i] + 1
        X.append(x_counter)
    print(X)
    print(len(X), 'x', len(X[0]), sep='')
