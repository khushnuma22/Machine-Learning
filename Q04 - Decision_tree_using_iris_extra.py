#Implement Decision-Tree Classifier on IRIS dataset.

import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
print("Dataset is below : ")
print(dataset)

X = dataset.data
y = dataset.target

#spliting into train and test set
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.33,random_state = 42)

print("Traing Features are : ")
print(X_train)
print("Training Labes are : ")
print(y_train)
print("Testion Features are : ")
print(X_test)
print("Testing Label are : ")
print(y_test)

# fitting train set
trclf = DecisionTreeClassifier()
trclf.fit(X_train,y_train)

# predict test set
dataClass = trclf.predict(X_test)
print("Accuracy Score : ",trclf.score(X_test,y_test))
tree.plot_tree(trclf)
plt.show()
