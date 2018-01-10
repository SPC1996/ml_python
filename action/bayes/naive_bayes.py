import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def binary_transform(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


def train(train_data, train_labels, class_num=10):
    prior_probability = np.zeros(class_num)
    conditional_probability = np.zeros((class_num, train_data.shape[1], 2))

    for i in range(len(train_labels)):
        img = binary_transform(train_data[i])
        label = train_labels[i]
        prior_probability[label] += 1
        for j in range(train_data.shape[1]):
            conditional_probability[label][j][img[j]] += 1

    for i in range(class_num):
        for j in range(train_data.shape[1]):
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]
            probability_0 = (float(pix_0) / float(pix_0 + pix_1)) * 1000000 + 1
            probability_1 = (float(pix_1) / float(pix_0 + pix_1)) * 1000000 + 1
            conditional_probability[i][j][0] = probability_0
            conditional_probability[i][j][1] = probability_1

    return prior_probability, conditional_probability


def calculate_probability(img, label, prior_probability, conditional_probability):
    probability = int(prior_probability[label])
    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])
    return probability


def predict(test_data, prior_probability, conditional_probability):
    predict = []
    predict_count = 1
    for img in test_data:
        print predict_count
        predict_count += 1
        img = binary_transform(img)
        max_label = 0
        max_probability = calculate_probability(img, 0, prior_probability, conditional_probability)
        for j in range(1, 10):
            probability = calculate_probability(img, j, prior_probability, conditional_probability)
            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)
    return np.array(predict)


if __name__ == '__main__':
    print 'start read data'

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values
    imgs = data[0::, 1::]
    labels = data[::, 0]
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323
    )
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'start training'
    prior_probability, conditional_probability = train(train_features, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'start predicting'
    test_predict = predict(test_features, prior_probability, conditional_probability)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print 'the accruacy score is ', score
