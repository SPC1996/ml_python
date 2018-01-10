import os
import time
import logging
import collections
import pandas as pd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn2pmml import PMMLPipeline, sklearn2pmml


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args, **kwargs)
        end_time = time.time()
        logging.debug('end %s(), cost %s seconds' % (func.__name__, end_time - start_time))
        return ret

    return wrapper


def trans_wav_to_csv(trans_type='stft', wav_path='chord', csv_path='data/data.csv'):
    path = wav_path
    for root, dir, files in os.walk(path):
        if root[len(wav_path) + 1:] != '':
            for file in files:
                print 'reading', root, '/', file
                y, sr = librosa.load(root + '/' + file, sr=44100)
                if trans_type == 'stft':
                    chroma_data = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
                elif trans_type == 'cqt':
                    chroma_data = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
                elif trans_type == 'cens':
                    chroma_data = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
                elif trans_type == 'mfcc':
                    chroma_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
                data_frame = pd.DataFrame(chroma_data).T
                data_frame[12] = root[len(wav_path) + 1:]
                data_frame.to_csv(csv_path, mode='a+', header=False, index=False)


def trans_wav_to_csv_2(wav_path, chord_type, csv_path, trans_type='stft'):
    y, sr = librosa.load(wav_path, sr=44100)
    if trans_type == 'stft':
        chroma_data = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    elif trans_type == 'cqt':
        chroma_data = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
    elif trans_type == 'cens':
        chroma_data = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
    elif trans_type == 'mfcc':
        chroma_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    data_frame = pd.DataFrame(chroma_data).T
    data_frame[12] = chord_type
    data_frame.to_csv(csv_path, mode='a+', header=False, index=False)


def draw(chroma_data, title):
    plt.figure()
    plt.subplot(1, 1, 1)
    librosa.display.specshow(chroma_data, y_axis='chroma', x_axis='time')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


@log
def to_pmml(classifier, pmml_path):
    pipeline = PMMLPipeline([
        ('classifier', classifier)
    ])
    sklearn2pmml(pipeline, pmml_path)


@log
def load_data(data_path):
    data = pd.read_csv(data_path, header=0).values
    features = data[:, 0:12].astype(np.float64)
    targets = data[:, 12]
    return features, targets


@log
def train(model, train_data, train_target):
    model.fit(train_data, train_target)
    return model


@log
def predict(model, test_data, test_target):
    return model.score(test_data, test_target)


@log
def full_pipeline(train_data_path, clf, clf_name='classifier'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    features, targets = load_data(train_data_path)
    train_data, test_data, train_target, test_target = train_test_split(
        features, targets, test_size=0.2, random_state=23323
    )
    clf = train(clf, train_data, train_target)
    score = predict(clf, test_data, test_target)
    print 'the score of ', clf_name, ' is ', score


@log
def full_pipeline_2(train_data_path, test_data_path, clf, clf_name='classifier'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_set, train_label = load_data(train_data_path)
    test_set, test_label = load_data(test_data_path)
    train(clf, train_set, train_label)
    test_score = predict(clf, test_set, test_label)
    print 'test_score is ', test_score


@log
def full_pipeline_3(train_data_path, pmml_path, clf, clf_name='classifier'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_set, train_label = load_data(train_data_path)
    clf = train(clf, train_set, train_label)
    to_pmml(clf, pmml_path)
    print 'the pmml of ', clf_name, ' export to ', pmml_path


@log
def cross_validation(train_data_path, clf, clf_name='classifier'):
    print 'cross_validation start...'
    train_set, train_label = load_data(train_data_path)
    train_score = cross_val_score(clf, train_set, train_label, scoring='accuracy', cv=10)
    print 'train_score is ', train_score
    print 'cross_validation end'


if __name__ == '__main__':
    cross_validation('data/data_cens.csv', SVC(kernel='linear'))
    # ordered_model = collections.OrderedDict()
    # ordered_model["RandomForestClassifier"] = RandomForestClassifier()
    # ordered_model["GradientBoostingClassifier"] = GradientBoostingClassifier()
    # ordered_model["LinearSVC"] = LinearSVC()
    # ordered_model["SVC(kernel='linear')"] = SVC(kernel='linear')
    # ordered_model["SVC(kernel='rbf')"] = SVC(kernel='rbf')
    # ordered_model["SVC(kernel='poly')"] = SVC(kernel='poly')
    # ordered_model["NuSVC(kernel='linear')"] = NuSVC(kernel='linear')
    # ordered_model["NuSVC(kernel='rbf')"] = NuSVC(kernel='rbf')
    # ordered_model["NuSVC(kernel='poly')"] = NuSVC(kernel='poly')
