from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot



# encoded_docs = [one_hot(d, 50) for d in X]
# X = pad_sequences(encoded_docs, maxlen=10, padding='post')

vocab = pd.read_csv('vocabs.txt', sep=',', header=None)
vocab.columns = ['word', 'id']

# words.head()

sentences = pd.read_csv('train-data.dat', sep=',', header=None)
sentences.head()

dict = {}                                       #creates dictionary from vocabs file
with open('vocabs.txt') as file:
    for line in file:
        (key, value) = line.split(', ')
        dict[key] = int(value)

com = []
for line in df.sentence:
    sen = ''
    for word in line.split():
      for id in vocab.id:
        if int(word) == int(id):
          sen = sen + ' ' + str(vocab.word[id]) 
    # print(sen)
    com.append(sen)

print(com)
