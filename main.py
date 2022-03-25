import prep
import predi
import pandas as pd

def main():
    X = prep.bow()
    # print(X)

    y = pd.read_csv('Data/train-label.dat', sep=',', header=None)
    y.columns = ['label']
    y = y[0:100]
    # print(y)
    predi.pred(X, y)

if __name__ == '__main__':
    main()


