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
    df = df[0:100]
    # print(df)

    dict = {}                                       #creates dictionary from vocabs file
    with open('Data/vocabs.txt') as file:
        for line in file:
            (key, value) = line.split(', ')
            dict[key] = int(value)

    # print(dict.values())

    # for (i, sentence) in enumerate(df['sentence']):
    #     print(sentence)

    # l = []
    # for j in range(len(df)):
    #     column = []
    #     for i in range(3):
    #         column.append(0)
    #     l.append(column)
    # print(l)

    X = []
    for data in df.sentence:
        v = []
        for w in data.split():
            k = []
            for j in range(8520):
                # print(j)
                if int(w) in dict.values():
                    # print(w)
                    # k[j] = 0
                    if j == w:
                        print(j)
                        k[j] = k[j] + 1
            v.append(k)
        X.append(v)
    print(X)
