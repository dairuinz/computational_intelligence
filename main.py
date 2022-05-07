import emb
import prep
import predi
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    df = pd.read_csv('Data/train-data.dat', sep=',', header=None)
    df.columns = ['sentence']
    df['sentence'] = df['sentence'].str.replace('<.*?>', '', regex=True)  # deletes <int>
    # print(df.head())
    df = df[0:100]
    X = df

    X = prep.bow(X)
    print('X shape: ', X.shape, sep='')
    # print(X.head(5))

    y = prep.out()
    print('y shape: ', y.shape, sep='')
    # print(y.dtypes)
    # print(y[0])

    com = emb.wpre(df)
    X = emb.wemb(com)
    emb.wpredi(X, y)

    # X = predi.scale(X)
    # # print(X)
    #
    # predi.pred(X, y)

if __name__ == '__main__':
    main()


