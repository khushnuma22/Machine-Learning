# Implementation of ANN algorithm using Iris


import sklearn as sk
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


dataset = datasets.load_iris()
print(dataset)
print(type(dataset))
X = dataset.data
y = dataset.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state = 0)
clf =  MLPClassifier(solver='sgd',alpha=0.00001,hidden_layer_sizes=(5,),max_iter=20) 
clf.fit(X_train,y_train)
dataClass_test = clf.predict(X_test)
dataClass_train = clf.predict(X_train)
print("The Iris Type is:")
print(dataset.target_names[dataClass_test])
print("score on train data:",accuracy_score(y_train,dataClass_train))
print("score on test data:",accuracy_score(y_test,dataClass_test))
print(confusion_matrix(y_test,dataClass_test,labels=[0,1,2]))
print("Weights : 0",clf.coefs_[0])
print("Weights : 1",clf.coefs_[1])
