## Project 1: Model Evaluation & Validation
### Predicting Boston Housing Prices
This document describes the implementation of a Machine Learning regressor that is capable of predicting Boston housing prices. The data used here is loaded in ([`sklearn.datasets.load_boston`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)) and comes from the StatLib library which is maintained at Carnegie Mellon University. You can find more information on this dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing) page.

#### Statistical analysis
* Total number of houses: 506
* Total number of features: 13
* Minimum house price: 5.0
* Maximum house price: 50.0
* Mean house price: 22.533
* Median house price: 21.2
* Standard deviation of house price: 9.188

#### Evaluating model performance
The problem of predicting the housing prices is not a classification problem since the numbers changing
with the time. So it is a **Regression problem** and uses regression problem's evaluation metrics
for model evaluation.

##### Measures of model performance
I think **Mean Squared Error(MSE)** is the most appropriate metric to use based on the following reasons:

* Predicting housing price problem is a regression problem since prices changes over time. So we cannot use Classification
  metrics such as 'Accuracy', 'Precision', 'Recall' and 'F1 Score'. Hence We need to choose between 'MSE' and 'MAE'.
* Between 'MSE' and 'MAE' both can work well with this problem but I would rather use 'MSE' due ti its properties. 'MSE'     penalizes larger errors more than smaller ones( since it is squarifies the absolute error so 0.2 will calc for 0.04 but    20 will be 40) and also it is a differentaible function.

##### Splitting the data
Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test.(*)

Hence to properly evaluate the model, the data we have must be split into two sets: a training set and a testing set to be able to:
  1- Give estimate on performance on independant datasets
  2- Serves as a check for overfitting

(*) Scikitlearn documentation.   

#### Installation
This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Udacity recommends our students install [Anaconda](https://www.continuum.io/downloads), i pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

#### Run

In a terminal or command window, navigate to the top-level project directory `boston_housing/` (that contains this README) and run one of the following commands:

  ```ipython notebook boston_housing.ipynb```

This will open the iPython Notebook software and project file in your browser.
