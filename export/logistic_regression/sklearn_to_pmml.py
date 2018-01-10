import pandas
from sklearn2pmml import PMMLPipeline
from sklearn2pmml.decoration import ContinuousDomain
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import sklearn2pmml

iris_df = pandas.read_csv("iris.csv")
train_data = iris_df[iris_df.columns.difference(["res"])]
target_data = iris_df["res"]

iris_pipeline = PMMLPipeline([
    ("mapper", DataFrameMapper([
        (["f_1", "f_2", "f_3", "f_4"], [ContinuousDomain(), Imputer()])
    ])),
    ("pca", PCA(n_components=3)),
    ("selector", SelectKBest(k=2)),
    ("classifier", LogisticRegression())
])
iris_pipeline.fit(train_data, target_data)

sklearn2pmml(iris_pipeline, "logistic_regression.pmml", with_repr=True)
