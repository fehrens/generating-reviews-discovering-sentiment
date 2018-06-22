import os
#import html
import numpy as np
import pandas as pd
import tensorflow as tf
import pkg_resources
import pickle
from pathlib import Path
from . import *
from . import encoder
from . import utils
from sklearn.linear_model import LogisticRegression

def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
        C=2**np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
    model.fit(trX, trY)
    #fe save the model 
    filename = ('log_reg.sav')
    pickle.dump(model, open(filename, 'wb'))
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return score, c, nnotzero

def predict_with_reg_cv(X):
    log_reg_file = Path('log_reg.sav')
    if log_reg_file.exists():
        print('LogReg file exists and will start the prediction now.')
        model = encoder.Model()
        Xt = model.transform(X)
        print('Done with NN and will start LogReg now.')
        log_reg_model = pickle.load(open(log_reg_file, 'rb'))
        prediction = log_reg_model.predict(Xt)
        print('Finished prediction and will return result now')
        return prediction
    else:    
        print('LogReg model does not exist and will train the model therefore. This might take a while')
        model = encoder.Model()
        trX, vaX, teX, trY, vaY, teY = sst_binary()
        trXt = model.transform(trX)
        vaXt = model.transform(vaX)
        teXt = model.transform(teX)
        full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
        print('%05.2f test accuracy'%full_rep_acc)
        print('%05.2f regularization coef'%c)
        print('%05d features used'%nnotzero)
        print('Done with learning and will call the prediction for the provided data')
        prediction = predict_with_reg_cv(X)
        return prediction

def get_logreg_model():
    log_reg_file = Path(pkg_resources.resource_filename('sentiment', 'model/log_reg.sav'))
    if log_reg_file.exists():
        print('LogReg file exists and will load model now')
        log_reg_model = pickle.load(open(log_reg_file, 'rb'))
        return log_reg_model
    else:
        print('LogReg file does not exist. You need to run predict_with_reg_cv once first.')
        return None    



def load_sst(path):
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir=pkg_resources.resource_filename('sentiment', 'data/')):
    """
    Most standard models make use of a preprocessed/tokenized/lowercased version
    of Stanford Sentiment Treebank. Our model extracts features from a version
    of the dataset using the raw text instead which we've included in the data
    folder.
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY


def find_trainable_variables(key):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))


def preprocess(text, front_pad='\n ', end_pad=' '):
    #text = html.unescape(text)  not working on my environment with python3, therefore disabled
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode()
    return text


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HParams(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
