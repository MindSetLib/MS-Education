from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import urllib
import urllib.request

# скачиваем википедию
urllib.request.urlretrieve("https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2", "enwiki-latest-pages-articles.xml.bz2")


# обучаем

with open('wiki.en.text', 'w') as fout:
    wiki = WikiCorpus(f, lemmatize=False, dictionary={})
    for i, text in enumerate(wiki.get_texts()):
        fout.write(' '.join(text) + '\n')
        if i == 99999:
            sys.exit()


model = Word2Vec(LineSentence('wiki.en.text'), size=200, window=5, min_count=3, workers=8)
# trim unneeded model memory = use (much) less RAM
model.init_sims(replace=True)
model.save('wiki.en.word2vec.model')


# тестируем модель
model.most_similar(’queen’, topn=3)


model.most_similar(positive=[’woman’, ’king’],
negative=[’man’], topn=2)


#-------------------------------------------------
# simplified example

import gzip
import gensim
import logging
import os
import urllib.request

# включаем логгирование
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# закачаем файл на диск
urllib.request.urlretrieve('https://github.com/kavgan/nlp-text-mining-working-examples/raw/master/word2vec/reviews_data.txt.gz', 'reviews_data.txt.gz')

def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break

# создадим чтение файла для получения данных

def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)

# файл на запись
data_file = os.path.join('reviews_data.txt.gz')

# создаем список списков для загрузки
documents = list(read_input(data_file))
logging.info("Done reading data file")


# загружаем данные в модель
# size - размер вектора, window окно, min_conts количество слов, workers количество потоков
model = gensim.models.Word2Vec(
    documents,
    size=150,
    window=10,
    min_count=2,
    workers=10)

# обучаем модель
model.train(documents, total_examples=len(documents), epochs=10)

w1 = "dirty"
print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

# look up top 6 words similar to 'polite'
w1 = ["polite"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'france'
w1 = ["france"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'shocked'
w1 = ["shocked"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# look up top 6 words similar to 'shocked'
w1 = ["beautiful"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))

# get everything related to stuff on the bed
w1 = ["bed", 'sheet', 'pillow']
w2 = ['couch']
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        negative=w2,
        topn=10))

# similarity between two different words
print("Similarity between 'dirty' and 'smelly'",
      model.wv.similarity(w1="dirty", w2="smelly"))

# similarity between two identical words
print("Similarity between 'dirty' and 'dirty'",
      model.wv.similarity(w1="dirty", w2="dirty"))

# similarity between two unrelated words
print("Similarity between 'dirty' and 'clean'",
      model.wv.similarity(w1="dirty", w2="clean"))


# поигрались

# проверим сохранение и открытие модели

model.save('model.bin')
# load model
new_model = gensim.models.Word2Vec.load('model.bin')


# пощупаем PCA
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

X = model[model.wv.vocab]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:20, 0], result[:20, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words[:20]):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

result = model.most_similar(positive=['woman', 'man'], negative=['sex'], topn=1)
print(result)

# в случае подготовки собственных данных для словаря необходимо перевести в кодировку utf

file_object  = open('from.txt', 'r')
file = open('to.txt', 'w',encoding='utf-8')

cnt=0
with file_object as fp:
    line = fp.readline()
    cnt = 1
    while line:
        file.write(line)
        line = fp.readline()
        cnt += 1

file.close()

# если работать через colab полезным будет загрузка данных с диска gdrive

from google.colab import drive
drive.mount('/content/gdrive')