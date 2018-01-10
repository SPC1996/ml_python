import pandas
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from sklearn.externals import joblib
from sklearn.svm import NuSVC


def load_model(load_model_filename):
    clf = joblib.load(load_model_filename)
    return clf


def dump(clf, dump_model_filename):
    joblib.dump(clf, dump_model_filename)


def load_data(load_data_filename):
    data = pandas.read_csv(load_data_filename)
    return data[data.columns.difference(["res"])].values, data["res"].values


def fit(clf, train_data, target_data):
    clf.fit(train_data, target_data)


def to_pmml(clf, pmml_name):
    pipeline = PMMLPipeline([
        ('classifier', clf)
    ])
    sklearn2pmml(pipeline, pmml_name)


if __name__ == '__main__':
    train_data, target_data = load_data('validate_data.csv')
    clf = NuSVC()
    clf.fit(train_data, target_data)
    dump(clf, 'NuSVC.pkl')
    to_pmml(clf, 'NuSVC_2.pmml')
