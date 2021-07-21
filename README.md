# KNN-Regression
KNN Regression Module (Python)

This KNN-Regression Model receives continuous numerical data as input to predict a single target output using KNN algorithm.

## Table of contents
* [Model Initialization](#model-init)
* [Model Fit](#fit)
* [Model Predict](#predict)
* [Model Check](#check)

## 1. Model Initialization (#model-init)
This model have multiple options to 
* Data Normalization
** On (Default)
** Off 
* Distance Calculation
** Euclidean
** Manhattan (Default)

Model initialization accepts two argument as the model setting
For example, if the desired model normalizes the input data before calculating the distance using the Manhattan Formula, then the model initialization can be written as
```
model = knn.knnreg("Manhattan", "On")
```

## 2. Model Fit (#fit)
Accepts a list of predictors and target into the model, normalizes the data (according to the option during initialization). The model will automatically fit the input data. 
The model accepts list as input, with each row referring to each sample. 
```
model.fit(a, p)
```
Where a is the predictor list, and p is the target list. 
This model can accept both single collumn target list, or a double-indexed list. 
```
Accepted1 = [1, 2, 3, 4, 5]
Accepted2 = [[1], [2], [3], [4], [5]]
```

## 3. Model Predict (#predict)
Predicts the output by calculating the distance between the input data and its neighbors. This function accepts the K value and the data to predict.
```
pv = model.predict(k, nd)
```
Where k is the K value, and nd is the input data used to predict. 

## 4. Model Check (#check)
Calculates the error and accuracy of a predicted value for validation purposes. Accepts the predicted value and the actual value. 
```
[error, accuracy] = model.check(pv, nt)
```
Where pv is the predicted value, and nt is the actual value. 



