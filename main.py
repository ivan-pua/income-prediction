"""
Name        : Qie Shang, Pua
zID         : z5157686
Course      : COMP3411 - Artificial Intelligence
University  : University of New South Wales

Goal -  To implement a Decision Tree model to predict whether an individual
        income exceeds $50K/yr based on census data.

Input data - Adult Data Set from the Machine Learning repository,
            http://archive.ics.uci.edu/ml/datasets/Adult
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

"""
This function transforms some of the categories into numbered labels to fit.  
Categories that are in string form such as "workclass" and "education" are converted into labels in integer form

Reason of doing this is because the inputs for classifier.fit(.....) must be integers 
"""
def transform (array):

    array = array.T
    length = len(array)
    # print(length)
    # print(array.shape[0])
    i = array.shape[0]
    j = array.shape[1]
    converted_array = np.empty((i, j))

    for row in range(length):

        if row is 0 or row is 2 or row is 4 or row is 10 or row is 11 or row is 12:
            # print("The number is " + array[row, 0])
            converted_array[row] = array[row]
            continue

        else:
            # if row is 7: print(array[row])
            temp = le.fit_transform(array[row])
            # if row is 7: print(temp)
            converted_array[row] = temp

    return converted_array.T


# Importing data
raw_data = pd.read_csv('adult.data', header=None, delimiter=",")
data = raw_data.to_numpy()

# Cleaning data by removing whitespaces
data = np.array([[item.strip() if type(item) is str else item for item in row] for row in data])

X = data[:, :-1]
y = data[:, -1]

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, shuffle = True)

# Preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# note that for many other classifiers, apart from decision trees,
# such as logistic regression or SVM, you would like to encode your categorical variables
# using One-Hot encoding. Scikit-learn supports this as well through the OneHotEncoder class.
# https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree

# Transformation
X_train = transform(X_train)
X_test = transform(X_test)

# Determining optimal accuracy by tuning hyperparameters
print("Entropy as criterion")
for i in range(1, 10):
    classifier = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    classifier = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Accuracy when the max depth is {i}: {metrics.accuracy_score(y_test, y_pred)}")

print("\nGini impurity as criterion")
for i in range(1, 10):
    classifier = DecisionTreeClassifier(max_depth=i) # default criteria is gini
    classifier = classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Accuracy when the max depth is {i}: {metrics.accuracy_score(y_test, y_pred)}")


"""
After tuning the parameters, it was decided that at max_depth = 3, the accuracy is considerably high (~0.84),
Since increasing the max_depth would not increase the accuracy significantly, 
the simplest hypothesis that fits the data is used which is consistent with Ockham's Razor.
"""
print("\nFinal")
classifier = DecisionTreeClassifier(max_depth=3, criterion="entropy") # if you change to 20, becomes random forest
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"The final accuracy of decision tree is: {metrics.accuracy_score(y_test, y_pred)}")

import graphviz
from sklearn import tree

# Exports the tree in png form
feature_cols = ['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
dot_data = tree.export_graphviz(classifier, out_file=None, feature_names= feature_cols, class_names=['<=50K', '>50K'])
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("decision-tree")


