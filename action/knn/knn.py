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


def predict(test_data, train_data, train_labels, k):
    predict = []
    count = 0
    for test_vec in test_data:
        print count
        count += 1

        knn_list = []
        max_index = -1
        max_dist = 0

        for i in range(k):
            label = train_labels[i]
            train_vec = train_data[i]
            dist = np.linalg.norm(train_vec - test_vec)
            knn_list.append((dist, label))

        for i in range(k, len(train_labels)):
            label = train_labels[i]
            train_vec = train_data[i]
            dist = np.linalg.norm(train_vec - test_vec)
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]
            if dist < max_dist:
                knn_list[max_index] = (dist, label)
                max_index = -1
                max_dist = 0

        class_total = 10
        class_count = [0 for i in range(class_total)]
        for dist, label in knn_list:
            class_count[label] += 1

        max_class = max(class_count)

        for i in range(class_total):
            if max_class == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)


if __name__ == '__main__':
    print 'start read data'

    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv', header=0)
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
    print 'knn do not need to training'
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'start predicting'
    test_predict = predict(test_features, train_features, train_labels, 15)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print 'the accruacy score is ', score
