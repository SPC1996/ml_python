import pandas
from sklearn2pmml import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml import sklearn2pmml

iris_df = pandas.read_csv("iris.csv")
train_data = iris_df[iris_df.columns.difference(["res"])]
target_data = iris_df["res"]

iris_pipeline = PMMLPipeline([
    ("classifier", DecisionTreeClassifier())
])
iris_pipeline.fit(train_data, target_data)

sklearn2pmml(iris_pipeline, "DecisionTreeIris.pmml", with_repr=True)
