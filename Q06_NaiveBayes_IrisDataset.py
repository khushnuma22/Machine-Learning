#Implement Naive Bayes Classification on IRIS dataset.

import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_iris()
print("Dataset is below : ")
print(dataset)

x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)
print("Traing Features are : ")
print(x_train)
print("Training Labes are : ")
print(y_train)
print("Testion Features are : ")
print(x_test)
print("Testing Label are : ")
print(y_test)

gnb = GaussianNB()
y_pred = gnb.fit(x_train,y_train).predict(x_test)

print('Number of mislabeled point out of a total %d points: %d' %(x_test.shape[0],(y_test != y_pred).sum()))

print('Accuracy Score: ', accuracy_score(y_test,y_pred))

print('Confusion Matrix: ', confusion_matrix(y_test,y_pred,labels = [0,1,2]))
