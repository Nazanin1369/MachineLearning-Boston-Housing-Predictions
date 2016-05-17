import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

def load_data():
    boston = datasets.load_boston()

    ## View data ##
    #print "Boston data:", boston.data
    #print "Boston keys:", boston.keys()
    #print "Boston feature names:", boston.feature_names
    #print "Boston description:", boston.DESCR
    #print "Boston target:", boston.target[:5]
    #print ''

    #pandas_dataframe = pd.DataFrame(boston.data)
    #pandas_dataframe.columns = boston.feature_names
    #pandas_dataframe['Price'] = boston.target
    #print pandas_dataframe.head()
    #print pandas_dataframe.query('Price > 20 & Price < 22')
    #print pandas_dataframe.query('RM > 8')
    #print pandas_dataframe
    return boston


def analye_data(city_data):
    '''Calculate the Boston housing statistics.'''

    #Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    #size of data
    print "Size of data:", (housing_features.shape[0])
    #Number of features?
    print "Number of features:", (housing_features.shape[1])
    #Minimum price
    print "Housing prices - minimum:", np.min(housing_prices)
    #Maximum price
    print "Housing prices - maximum:", np.max(housing_prices)
    #Calculate the mean
    print "Housing prices - mean:", np.mean(housing_prices)
    # Calculate median?
    print "Housing prices - median:", np.median(housing_prices)
    # Calculate standard deviation?
    print "Housing prices - standard deviation:", np.std(housing_prices)



def main():
    #Load data
    city_data = load_data();

    #do statistical analysis on the data
    analye_data(city_data);



main()
