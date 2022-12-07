#######################################################
###NINTH SCRIPT - LINEAR REGRESSIONS LEARNING CURVES###
#######################################################

import pandas as pd
import numpy as np
from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
COMMENTS:
    
This is the ninth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

In this last set of codes, we produce the learning curves.

We start here with producing the numbers in Figure 5, 
"Learning Curve of Linear Regression on the Original Data".

The train-test splits, since changing in thier size, are redone everytime.
'''

def linreg_train_test(X_train, y_train, X_test, y_test, 
                      sample_weights = None):
    
    lineareg = LinearRegression()

    lineareg_fitted = lineareg.fit(X_train, y_train, sample_weight = sample_weights)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_train)
    
    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()
    
    list_of_results = [Mse_lineareg_test, Mse_lineareg_train]
    
    return list_of_results

path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits\\'

dest_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\'

#It is necessary to just import one train - test split, since we will bind
#them back together to obtain the full dataset and on that compute everytime
#a new split with different training size, as in the Learning Curves' figures.

train_1 = pd.read_csv(path + 'train_1.csv')

test_1 = pd.read_csv(path + 'test_1.csv')

most_pop_cats = ['bd7ms_1', 'b7accom_1',
                 'b7ten2_2', 'bd7ecact_1',
                 'bd7hq13_3', 'b7khldl2_2',
                 'b7khllt_2', 'bd7bmigp_2',
                 'bd7smoke_0', 'bd7dgrp_1',
                 'bd7maliv_1', 'bd7paliv_1']

least_pop_cats = ['b7accom_4', 'bd7ecact_-8', 'bd7ecact_7', 'b7khldl2_-8',
                  'bd7hq13_-8', 'bd7ms_6', 'bd7ms_-7', 'bd7maliv_-8',
                  'bd7smoke_6', 'bd7smoke_-7', 'b7ten2_6', 'b7ten2_-8']

least_most = most_pop_cats + least_pop_cats

train_1.drop(least_most, axis = 1, inplace = True)

test_1.drop(least_most, axis = 1, inplace = True)

train_1['Physical Health'] = train_1['Physical Health'].apply(lambda x: -1 * x)

test_1['Physical Health'] = test_1['Physical Health'].apply(lambda x: -1 * x)

train_sizes = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06,
               0.05, 0.04, 0.03, 0.02, 0.01] 

np.random.seed(42)

##################
##ORIGINAL MODEL##
##################

y_train_1_orig = train_1['lifesatisfaction']
    
X_train_1_orig = train_1[['Log Income', 'Educational Achievement', 'Employed', 
                        'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                        'Emotional Health']]
    
y_test_1_orig = test_1['lifesatisfaction']
    
X_test_1_orig = test_1[['Log Income', 'Educational Achievement', 'Employed', 
                      'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                      'Emotional Health']]

X_1_orig = pd.concat([X_train_1_orig, X_test_1_orig], axis = 0) 

X_1_orig.reset_index(drop = True, inplace = True)

y_1_orig = pd.concat([y_train_1_orig, y_test_1_orig], axis = 0) 

y_1_orig.reset_index(drop = True, inplace = True)

MSEs_test_orig_stand = []

train_size = []

for i in train_sizes:
    
    seed = randint(0, 1000)

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_1_orig, 
                                                                            y_1_orig,
                                                                            test_size = 1 - i,
                                                                            random_state = seed)

    
    train_test_linreg_orig_stand = linreg_train_test(X_train = X_train_orig, 
                                                     y_train = y_train_orig, 
                                                     X_test = X_test_orig, 
                                                     y_test = y_test_orig)
            
    MSEs_test_orig_stand.append(train_test_linreg_orig_stand[0])
        
    train_size.append(i)
    
MSEs_and_train_size = pd.DataFrame([MSEs_test_orig_stand, train_size]).T

MSEs_and_train_size.columns = ['Test MSE', 'train size (%)']

MSEs_and_train_size['train size (%)'] = MSEs_and_train_size['train size (%)'] * 100

MSEs_and_train_size.to_csv(dest_path + 'Linreg_orig_learning_curve.csv')



