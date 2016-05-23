## Project 1: Model Evaluation & Validation

[View the results online](https://nazanin1369.github.io/MachineLearning-Boston-Housing-Predictions/)

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
  * Give estimate on performance on independant datasets
  * Serves as a check for overfitting

(*) Scikitlearn documentation.   

##### Cross-Validation
Even if we split the data, our knowledge while tuning a model's parameters can add biases to the model, which can still be overfit to the test data. Therefore, ideally we need a third set the model has never seen to truly evaluate its performance. The drawback of splitting the data into a third set for model validation is that we lose valuable data that could be used to better tune the model.

An alternative to separating the data is to use cross-validation. Cross-validation is a way to predict the fit of a model to a hypothetical validation set when such a set is not explicitly available. There is a variety of ways in which cross-validation can be done. These methods can be divided into two groups: exhaustive cross-validation, and non-exhaustive cross-validation. Exhaustive methods are methods which learn and test on all possible ways to divide the original data into a training and a testing set. As such, these methods can take a while to compute, especially as the amount of data increases. Non-exhaustive methods, as the name says, do not compute all ways of splitting the data.

**K-fold cross-validation** in an example of exhaustice method, which consists of randomly partitioning the data into k equal sized subsets. Of these, one subset becomes the validation set, while the other sets are used for training the model, and the process is executed k times, one for each validation subset. The performance of the model, then, is the average of the performance of model in each of the k executions. This method is attractive because all data points are used in the overall process, with each point used only once for validation.


##### Grid Search: Searching for estimator parameters
Machine learning models are basically mathematical functions that represent the relationship between different aspects of data. Models can have parameters some can be learnt during the trainig phase ans some other which called **hyperparameters** must be specified outside the training procedures such as decision trees depth ot number of leaves.
This type of hyperparameter controls the capacity of the model, i.e., how flexible the model is, how many degrees of freedom it has in fitting the data. Proper control of model capacity can prevent **overfitting**, which happens when the model is too flexible, and the training process adapts too much to the training data, thereby losing predictive accuracy on new test data. So a proper setting of the hyperparameters is important.

Grid search is one of the algorithms used for tuning hyperparameters; true to its name, picks out a grid of hyperparameter values, evaluates every one of them, and returns the winner. For example, if the hyperparameter is the number of leaves in a decision tree, then the grid could be 10, 20, 30, …, 100. For regularization parameters. Some guess work is necessary to specify the minimum and maximum values. So sometimes people run a small grid, see if the optimum lies at either end point, and then expand the grid in that direction. This is called manual grid search.

Grid search is dead simple to set up and trivial to parallelize. It is the most expensive method in terms of total computation time. However, if run in parallel, it is fast in terms of wall clock time.

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
