from re import sub
from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')


def remove_apostrophe(sentence):
    sentence = sub(r"can[’']t", 'can not', sentence)
    sentence = sub(r"won[’']t", "will not", sentence)
    sentence = sub(r"shan[’']t", "shall not", sentence)
    sentence = sub(r"[’']s", ' is', sentence)
    sentence = sub(r"n[’']t", ' not', sentence)
    sentence = sub(r"i[’']m", 'i am', sentence)
    sentence = sub(r"[’']ve", ' have', sentence)
    sentence = sub(r"[’']re", " are", sentence)
    sentence = sub(r"[’']d", ' would', sentence)
    sentence = sub(r"[’']ll", ' will', sentence)
    return sentence


def remove_stopwords(sentence):
    return ' '.join([word for word in sentence.split(' ') if word not in eng_stopwords])
