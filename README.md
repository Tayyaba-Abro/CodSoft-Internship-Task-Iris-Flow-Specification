# CodSoft Internship Task: Iris Flow Specification

## Introduction

This project involves exploring the Iris dataset and building machine learning models to classify Iris flowers into different species based on their sepal and petal measurements. It comprises three distinct species of Iris flowers: setosa, versicolor, and virginica. These species, each possessing unique characteristics, can be accurately distinguished based on their vital statistics.

## Quick Link
Dataset: [Iris Flower Classification Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)

## Project Steps

### Step 1: Importing Libraries
```python
# import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

### Step 2: Reading Data
```python
# read data
df = pd.read_csv("IRIS.csv")
df
```

![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/e3bbd75f-c37c-46ba-8060-f3275d7c4bcf)

### Step 3: Preprocessing
```python
# figure out the number of rows and columns in the dataframe
df.shape
```
![shape](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/0b22120a-8b46-4357-8865-938607a95f95)

```python
# check out info on the columns of the dataset
df.info()
```
![info](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/8cff88b2-2379-4a8f-b888-2267abc03ee0)

```python
# check for any missing values
df.isna().sum()
```
![null values](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/aa042d39-204a-41bc-85cb-2059a1c21309)

```python
# show statistics of the dataframe
df.describe()
```
![describe](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/dbc53c75-da3a-4f7e-9859-1d0b589cd6d9)

```python
# replace strings with numericals for simplicity in model training
df["species"].replace({"Iris-setosa":0 , "Iris-versicolor":1 , "Iris-virginica":2} , inplace = True)
df
```
![change](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/48dc9692-8107-4906-b5c8-4777102f9e95)

### Step 4: Explonatory Data Analysis (EDA)

#### i. Species Measurements Correlation 
```python
# plotting correlation heatmap
dataplot = sns.heatmap(df.corr(), cmap="RdPu", annot=True)
# displaying heatmap
plt.show()
```
![correlation](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/5ff7fa37-9b73-4cde-862f-27b1595f0b48)

#### ii. Pairwise Feature Correlation (Excluding Species)
```python
# show the relation between each column excep species which lies on top of each visualization
plt.figure(figsize=(4, 3))
sns.pairplot(df, hue='species')
plt.show()
```
![pairwise](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/365bcd2f-aa9a-43a3-8168-1ba740f84556)

#### iii. Sepal Length Distribution Analysis
```python
# create histogram of sepal_length with species on top of it
sns.histplot(data=df, x='sepal_length', kde=True, hue='species')
plt.show()
```
![sepal length graph](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/5ab9b019-8463-41fb-b268-ccc4f6b0beab)

```python
# create box plot to show distribution of data in sepal_length column
sns.boxplot(data=df, x='sepal_length', color = 'Purple')
plt.show()
```
![sepal length data](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/d26b1ad6-2a70-4615-a691-83038d4c2dda)

#### iv. Sepal Width Distribution Analysis
```python
# create histogram of sepal_width with species on top of it
sns.histplot(data=df, x='sepal_width', kde=True, hue='species')
plt.show()
```
![sepal width graph](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/fb25cb84-5931-4304-87d2-cd01758d4fee)

```python
# create box plot to show distribution of data in sepal_width column
sns.boxplot(data=df, x='sepal_width', color = 'Purple')
plt.show()
```
![sepal width data](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/56dcfed5-5c4e-48e8-875d-ab4f9b5f2452)

#### v. Petal Length Distribution Analysis
```python 
# create histogram of petal_length with species on top of it
sns.histplot(data=df, x='petal_length', kde=True, hue='species')
plt.show()
```
![patel length](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/dd7fd4c3-b3a7-4bae-838d-1693010e0a41)

```python
# create box plot to show distribution of data in petal_length column
sns.boxplot(data=df, x='petal_length', color = 'Purple')
plt.show()
```
![petal length data](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/4cb8c7b7-63b2-4133-be7d-d035a5ffb605)

#### vi. Petal Width Distribution Analysis
```python
# create histogram of petal_width with species on top of it
sns.histplot(data=df, x='petal_width', kde=True, hue='species')
plt.show()
```
![petal width](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/1002c016-8dac-4ba1-a251-326f7bae17a8)

```python
# create box plot to show distribution of data in petal_width column
sns.boxplot(data=df, x='petal_width', color = 'Purple')
plt.show()
```
![petal width data](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/e9d3459d-c853-4ef8-85af-fc39c4f346a9)

#### vii. Count of Species in Dataframe 
```python
# show histogram to show how many entries lie for each species in the dataframe
custom_palette = ["Pink", "Purple", "Gray"]
sns.countplot(x = df['species'] , data = df, palette = custom_palette)
```
![species](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/6a3a2e40-f0f2-4460-b68f-5ae8a731e3a4)

### Modelling Techniques
we explore various machine learning models to classify Iris flowers based on their sepal and petal measurements. Each of the following techniques offers a unique approach to this classification task: 

```python
# creating target and learning variables
X = df[["sepal_length" , "sepal_width" , "petal_length" , "petal_width"]]
y = df["species"]

# splitting the data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
```

#### i. Linear Regression
```python
LR = LogisticRegression(max_iter=200)
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic regression accuracy: {:.2f}%'.format(LRAcc*100))
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/e698098e-77a2-46d9-b2d1-d26f496e40e5)

#### ii. K Nearest Neighbors
```python
# Creating lists to store accuracy scores for different n_neighbors values
k_range = range(1, 51)
accuracy_scores = []

# Iterate through different values of n_neighbors
for k in k_range:
    # Create a KNN classifier with the current value of n_neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the classifier to the training data
    knn_classifier.fit(X_train, y_train)
    
    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)
    
    # Calculate the accuracy score and append it to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(12,8))
# create a line graph for showing accuracy score (scores_list) for respective number of neighbors used in the KNN model
plt.plot(k_range, accuracy_scores, linewidth=2, color='purple')
# values for x-axis should be the number of neighbors stored in kRange
plt.xticks(k_range)
plt.xlabel('Neighbor Number')
plt.ylabel('Accuracy Score of KNN')
plt.show()
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/74a13bee-3676-4bc0-8340-b47042c87c05)

```python
# Find the best value of n_neighbors with the highest accuracy
best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
best_accuracy = max(accuracy_scores)

print(f"Best number of neighbors: {best_k}")
print(f"Highest accuracy: {best_accuracy}")
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/6c23fb06-3aba-4bad-8ae5-f420ecbf2f1d)

```python
# Creating a KNN model with best parameters i.e., number of neighbors = 23
classifier_knn = KNeighborsClassifier(n_neighbors = 1)

# fit training data to the KNN model
classifier_knn.fit(X_train,y_train)
# evaluate test data on the model
pred = classifier_knn.predict(X_test)
# show regression score
KNNAcc = accuracy_score(y_test, pred)
print('K-Nearest Neighbors Classifier accuracy: {:.2f}%'.format(KNNAcc*100))
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/452e4d90-5b0d-41b7-82dc-a3627b4e76ba)

##### iii. Decision Tree Classifier
```python
# create a decision tree classifier object
dt = DecisionTreeClassifier()
# train the model on the training dataset
dt.fit(X_train, y_train)
# make predictions on the test data
y_pred = dt.predict(X_test)
# calculate accuracy
DTCAcc = accuracy_score(y_test, y_pred)
print('Decision Tree Classifier accuracy: {:.2f}%'.format(KNNAcc*100))
# create a decision tree classifier object
dt = DecisionTreeClassifier()
# train the model on the training dataset
dt.fit(X_train, y_train)
# make predictions on the test data
y_pred = dt.predict(X_test)
# calculate accuracy
DTCAcc = accuracy_score(y_test, y_pred)
print('Decision Tree Classifier accuracy: {:.2f}%'.format(KNNAcc*100))
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/b16c7098-c4af-4f5c-8bc5-362542f75b14)

#### iv. Random Forest Classifier
```python
# create a decision tree classifier object
rf = RandomForestClassifier()
# train the model on the training dataset
rf.fit(X_train, y_train)
# make predictions on the test data
y_pred = rf.predict(X_test)
# calculate accuracy
RFCAcc = accuracy_score(y_test, y_pred)
print('Random Forest Classifier accuracy: {:.2f}%'.format(RFCAcc*100))
```
![image](https://github.com/Tayyaba-Abro/CodSoft-Internship-Task-Iris-Flow-Specification/assets/47588244/01e3d5d1-c24f-4f46-8a0f-aabeb3f2683f)

For each technique, we evaluate its accuracy and performance in classifying Iris flowers into their respective species. Let's see how these models stack up against each other!

## Conclusion
This project revealed valuable insights into the Iris dataset, and we've achieved an accuracy ratio of 79.74%. This demonstrates the effectiveness of our chosen modelling techniques.












