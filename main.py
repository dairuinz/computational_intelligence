import emb
import prep
import predi
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    df = pd.read_csv('Data/train-data.dat', sep=',', header=None)       #reads input data
    df.columns = ['sentence']           #names column of data
    df['sentence'] = df['sentence'].str.replace('<.*?>', '', regex=True)    #deletes <int>
    # print(df.head())
    df = df[0:100]
    X = df      #input data

    y = prep.out()      #output data preprocessing
    # print('y shape: ', y.shape, sep='')

    answer = input('Select method:\n\t 1. Bag of Words\n\t '
                   '2. Word Embeddings\n')

    if answer == '1':
        X = prep.bow(X)      # input bag of words data preprocessing
        # print('X shape: ', X.shape, sep='')
        X = predi.scale(X)      #scaling of input data
        predi.pred(X, y)        #model
    elif answer == '2':
        com = emb.wpre(X)       #word embeddings preparation
        X = emb.wemb(com)       #word2vec implementation
        emb.wpredi(X, y)        #model


if __name__ == '__main__':
    main()


