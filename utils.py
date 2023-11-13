import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# Dl package of the pretrain tokenizer
nltk.download("punkt")

# Initialize stemmer
stemmer = PorterStemmer()

# Tokenizes each sentence using a pretrain tokenizer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Lowers and stems words
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_tokens):
    tokenized_sentence = [stem(token) for token in tokenized_sentence]

    bag = np.zeros(len(all_tokens), dtype=np.float32)

    for i in range(len(all_tokens)):
        if all_tokens[i] in tokenized_sentence:
            bag[i] = 1.0
    return bag
    
