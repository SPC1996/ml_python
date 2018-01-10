# encoding=utf-8
import time
import math
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000
        self.w = [0.0]

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        exp_wx = math.exp(wx)
        predict_1 = exp_wx / (1 + exp_wx)
        predict_0 = 1 / (1 + exp_wx)
        if predict_1 > predict_0:
            return 1
        else:
            return 0

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        times = 0
        while times < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index]
            if y == self.predict_(x):
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            times += 1
            correct_count = 0
            wx = sum([self.w[i] * x[i] for i in range(len(self.w))])
            exp_wx = math.exp(wx)
            for i in range(len(self.w)):
                self.w[i] -= self.learning_step * (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':
    time_1 = time.time()

    print 'start load data'
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[:, 0]
    train_data, test_data, train_target, test_target = train_test_split(imgs, labels, test_size=0.33,
                                                                        random_state=123456)
    time_2 = time.time()
    print 'load data cost ', time_2 - time_1, ' seconds\n'

    print 'start training'
    lr = LogisticRegression()
    lr.train(train_data, train_target)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' seconds\n'

    print 'start predicting'
    predict_target = lr.predict(test_data)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' seconds\n'

    score = accuracy_score(test_target, predict_target)
    print 'the accuracy score is ', score
