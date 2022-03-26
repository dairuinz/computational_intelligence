import prep
import predi
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    X = prep.bow()
    print('X shape: ', X.shape, sep='')
    # print(X)

    y = prep.out()
    print('y shape: ', y.shape, sep='')
    # print(y.dtypes)

    X = predi.scale(X)

    predi.pred(X, y)        

if __name__ == '__main__':
    main()


