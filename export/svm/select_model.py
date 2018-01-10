import collections
import pandas
from sklearn2pmml import PMMLPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier


def calculate_accuracy(source_data, target_data):
    true_data_num, false_data_num = 0.0, 0.0
    for i, j in zip(source_data, target_data):
        if i == j:
            true_data_num += 1
        else:
            false_data_num += 1
    accuracy = true_data_num * 100 / (true_data_num + false_data_num)
    return str(accuracy) + '%'


def train(train_data_path, classifier):
    train_df = pandas.read_csv(train_data_path)
    train_data = train_df[train_df.columns.difference(["res"])]
    target_data = train_df["res"]
    pipeline = PMMLPipeline([
        ("classifier", classifier)
    ])
    pipeline.fit(train_data, target_data)
    return pipeline


def validate(validate_data_path, pipeline):
    validate_df = pandas.read_csv(validate_data_path)
    validate_data = validate_df[validate_df.columns.difference(["res"])]
    target_data = validate_df["res"]
    return pipeline.predict(validate_data), target_data


def test(model_dict, train_data_path, validate_data_path):
    for key in model_dict:
        m_pipeline = train(train_data_path, model_dict[key])
        result_data, success_data = validate(validate_data_path, m_pipeline)
        accuracy = calculate_accuracy(result_data, success_data)
        print("The accuracy of model '%s' is %s" % (key, accuracy))


if __name__ == '__main__':
    ordered_model = collections.OrderedDict()
    ordered_model["KNeighborsClassifier"] = KNeighborsClassifier()
    ordered_model["RadiusNeighborsClassifier"] = RadiusNeighborsClassifier()
    ordered_model["DecisionTreeClassifier"] = DecisionTreeClassifier()
    ordered_model["ExtraTreeClassifier"] = ExtraTreeClassifier()
    ordered_model["ExtraTreesClassifier"] = ExtraTreesClassifier()
    ordered_model["RandomForestClassifier"] = RandomForestClassifier()
    ordered_model["GradientBoostingClassifier"] = GradientBoostingClassifier()
    ordered_model["LinearSVC"] = LinearSVC()
    ordered_model["SVC(kernel='linear')"] = SVC(kernel='linear')
    ordered_model["SVC(kernel='rbf')"] = SVC(kernel='rbf')
    ordered_model["SVC(kernel='poly')"] = SVC(kernel='poly')
    ordered_model["NuSVC(kernel='linear')"] = NuSVC(kernel='linear')
    ordered_model["NuSVC(kernel='rbf')"] = NuSVC(kernel='rbf')
    ordered_model["NuSVC(kernel='poly')"] = NuSVC(kernel='poly')
    ordered_model["GaussianProcessClassifier(multi='one_vs_one')"] = GaussianProcessClassifier(multi_class='one_vs_one')
    ordered_model["GaussianProcessClassifier(multi='one_vs_rest')"] = GaussianProcessClassifier(
        multi_class='one_vs_rest')

    test(ordered_model, "train_data.csv", "validate_data.csv")
