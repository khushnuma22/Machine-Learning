# KNN using iris dataset

#Step-1 Import all library
import sklearn as sk
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Step-2 Load Dataset
dataset = datasets.load_iris()
print("Dataset is below : ")
print(dataset)
#Step-4 Assign the data and target value
x = dataset.data
y = dataset.target
#Step-5 split train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
print("Traing Features are : ")
print(x_train)
print("Training Labes are : ")
print(y_train)
print("Testion Features are : ")
print(x_test)
print("Testing Label are : ")
print(y_test)
#Step-6 Model define
model = sk.neighbors.KNeighborsClassifier(n_neighbors = 12, weights = 'distance', metric = 'euclidean')
#Step-7 Model fit
model.fit(x_train, y_train)
#Step-8 Model test(Predictions)
dataclass = model.predict(x_test)
print("Dataclass is below : ")
print(dataclass)
print("Name of target Features : ")
print(dataset.target_names[dataclass])
#Step-9 Accuracy Score
print("Accuracy Score : ")
print(accuracy_score(y_test, dataclass, normalize = True, sample_weight = None))
#Step-10 Confusion Matrix
print("Confusion Matrix : ")
print(sk.metrics.confusion_matrix(y_test, dataclass, labels = [0, 1, 2]))
