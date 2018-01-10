import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_hog_features(data_set):
    features = []
    hog = cv2.HOGDescriptor('../data/hog.xml')
    for img in data_set:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)
    features = np.array(features)
    features = np.reshape(features, (-1, 324))
    return features


def train(train_data, train_labels, separate_num=0, study_step=1):
    train_data_size = len(train_labels)
    w = np.zeros((train_data.shape[1], 1))
    b = 0

    study_count = 0
    nochange_count = 0
    nochange_upper_limit = 30000

    while True:
        nochange_count += 1
        if nochange_count > nochange_upper_limit:
            break

        index = random.randint(0, train_data_size - 1)
        img = train_data[index]
        label = train_labels[index]

        yi = int(label != separate_num) * 2 - 1
        result = yi * (np.dot(img, w) + b)

        if result <= 0:
            img = np.reshape(train_data[index], (train_data.shape[1], 1))
            w += img * yi * study_step
            b += yi * study_step
            study_count += 1
            if study_count > train_data.shape[0]:
                break
            nochange_count = 0
    return w, b


def predict(test_data, w, b):
    predict = []
    for img in test_data:
        result = np.dot(img, w) + b
        result = result > 0
        predict.append(result)
    return np.array(predict)


if __name__ == '__main__':
    print 'start read data'

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    features = get_hog_features(imgs)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.33, random_state=23323
    )
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'start training'
    w, b = train(train_features, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'start predicting'
    test_predict = predict(test_features, w, b)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print 'the accruacy score is ', score
