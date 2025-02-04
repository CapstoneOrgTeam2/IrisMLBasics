import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pandas import plotting, DataFrame
from sklearn.neighbors import KNeighborsClassifier


# This code is based off this lesson:
# https://arcca.github.io/An-Introduction-to-Machine-Learning-Applications/03-scikit-learn-iris-dataset/index.html

# More info on the Iris dataset can be found here:
# https://archive.ics.uci.edu/dataset/53/iris

iris_dataset = load_iris()

# Uncomment to see details about the iris dataset
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#
# print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# print("\n")
# print("Targets:\n{}".format(iris_dataset['target'][:]))
# print("\n")
# print("Target names:\n{}".format(iris_dataset['target_names']))
# print("\n")
# print("Feature names:\n{}".format(iris_dataset['feature_names']))
# print("\n")
# print("Dataset location:\n{}".format(iris_dataset['filename']))
#
# print(iris_dataset['DESCR'])

X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataframe = DataFrame(X_train, columns=iris_dataset.feature_names)
grr = plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.show()     # Behold the scatter matrix, useful to visualize the pairwise relationships between features.
# Note since the classes seem well separated, a ML model will be able to learn to separate them.


