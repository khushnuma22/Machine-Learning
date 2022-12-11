# KNN using BreastCancerDataset Dataset

#Step-1 Import all library
import sklearn as sk
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("BreastCancerDataset.csv")
print("Dataset is below : ")
print(dataset)
print("Dataset Information : ")
print(dataset.info())
print("\n-----Head-----")
print(dataset.head())
print("\n-----Describe-----")
print(dataset.describe())
print("\n-----Shape-----")
print(dataset.shape)

#Decide Dependent & Independent Attributes
x = dataset[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
              "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]]
y = dataset[["diagnosis"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
print("Traing Features are : ")
print(x_train)
print("Training Labes are : ")
print(y_train)
print("Testion Features are : ")
print(x_test)
print("Testing Label are : ")
print(y_test)
model = sk.neighbors.KNeighborsClassifier(n_neighbors = 12, weights = 'distance', metric = 'euclidean')
model.fit(x_train, y_train.values.ravel()) #.values will give the values in a numpy array .ravel will convert that array shape to (n, )

dataclass = model.predict(x_test)
print("Accuracy Score : ")
print(accuracy_score(y_test, dataclass, normalize = True, sample_weight = None))

print("Confusion Matrix : ")
print(confusion_matrix(y_test, dataclass))
