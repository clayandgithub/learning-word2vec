# import modules & set up logging
import gensim, logging
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def simple_test() :
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, min_count=1)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

def dir_test():
    sentences = MySentences('./text_dataset') # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)
    model.save("./model")

def similarity_test():
    model = gensim.models.Word2Vec.load('./model')
    print(model.most_similar(positive=['woman', 'queen'], negative=['man'], topn = 2))
    print(model.similarity('woman', 'man'))
    print(model.doesnt_match("woman man queen king".split()))

def raw_output():
    model = gensim.models.Word2Vec.load('./model')
    print(model['woman'])

def test_similar_by_vec():
    model = gensim.models.Word2Vec.load('./model')
    woman_vec = model['woman']
    result = model.most_similar(positive=[woman_vec], topn = 4)
    # result = model.similar_by_vector(woman_vec, topn = 4)
    print(result[0][0])

def test_similar_by_word():
    model = gensim.models.Word2Vec.load('./model')
    word = 'fake'
    if word in model.vocab:
        print model.most_similar(positive=[word], topn = 4)
    else:
        print('\'' + word + '\' is not in the vacabulary!')

test_similar_by_word()