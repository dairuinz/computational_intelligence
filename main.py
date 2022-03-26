import prep
import predi
import pandas as pd

def main():
    X = prep.bow()
    print('X shape: ', X.shape, sep='')

    y = prep.out()
    print('y shape: ', y.shape, sep='')
    # print(y.dtypes)

    predi.pred(X, y)

if __name__ == '__main__':
    main()


