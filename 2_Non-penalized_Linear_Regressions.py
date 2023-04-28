########################################
###SECOND SCRIPT - LINEAR REGRESSIONS###
########################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

'''
COMMENTS:

This is the second script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

Here, in particular, we proceed with running the non-penalized
linear regressions on the 100 train-test splits generated in the
previous scripts.
'''

#We create a custom function to fit and predict with linear regression.
#The sklearn's class LinearRegression() has as default fit_intercept = True,
#hence we don't need to add a constant column to our data. 

def linreg_train_test(X_train, y_train, X_test, y_test):
    
    lineareg = LinearRegression()

    lineareg_fitted = lineareg.fit(X_train, y_train)
    
    lineareg_yhat_test = lineareg_fitted.predict(X_test)

    Mse_lineareg_test = ((lineareg_yhat_test - y_test)**2).mean()
    
    lineareg_yhat_train = lineareg_fitted.predict(X_train)
    
    Mse_lineareg_train = ((lineareg_yhat_train - y_train)**2).mean()
    
    list_of_results = [Mse_lineareg_test, Mse_lineareg_train]
    
    return list_of_results

path = 'C:\\some_local_path'  

results_path = 'C:\\some_other_local_path'

#Each of the training and test sets is imported as a 
#pandas dataframe into a list. In this way, it will be easier
#to perform all the operatiosn on each of the 100 training and test
#sets in loop. 

#Importing training sets

training_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + '\\train_' + j + '.csv'

    training_sets.append(pd.read_csv(import_path))

#training_sets is the list with the 100 training sets,
#each of them a 7093 x 77 pandas dataframe. 

#Importing test set

test_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + '\\test_' + j + '.csv'
        
    test_sets.append(pd.read_csv(import_path))
    
#test_sets is the list with the 100 training sets,
#each of them a 1774 x 77 pandas dataframe. 

#As indicated in the paper, in running the non-penanlized
#regressions, for each of the categorical variables transformed
#in dummies (one-hot-encoding) we not only dropped the most 
#populous category - as customary - but also all those categories
#with at most 15 individuals - they're avaialable in Appendix D.

#As explained in the second footnote in the paper, this deletion
#was necessary to avoid 18 perfectly multicollinear cases,
#since all of the 1’s were randomly-allocated to the test set 
#(producing a column of 0’s in the training set, hence perfectly collinear).

#As we argue in the paper, this necessary procedure may 
#nonetheless still induce some loss of information, overcome
#instead by Penalized Regressions and Random Forests. 

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

#Keep in mind that, clearly:
#len(training_sets) == len(test_sets) = 100
#hence they're interchangeable in defining the iteration range.

for i in range(0, len(training_sets)):
    
    training_sets[i].drop(least_most, axis = 1, inplace = True)
    
    test_sets[i].drop(least_most, axis = 1, inplace = True)
    
#We invert the values of physical health to represent 
#good rather than bad health.

for i in range(0, len(training_sets)):
    
    training_sets[i]['Physical Health'] = training_sets[i]['Physical Health'].apply(lambda x: -1 * x)
    
    test_sets[i]['Physical Health'] = test_sets[i]['Physical Health'].apply(lambda x: -1 * x)
    
#########################
###ORIGINAL 8 FEATURES###
#########################

#The test and train MSE for each of the 100 fitted linregs
#is saved in its respective list, created below.

MSEs_test_orig = []

MSEs_train_orig = []

for i in range(0, len(training_sets)):
    
    y_train_orig = training_sets[i]['lifesatisfaction']
    
    X_train_orig = training_sets[i][['Log Income', 'Educational Achievement', 'Employed', 
                                     'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                                     'Emotional Health']]
    
    y_test_orig = test_sets[i]['lifesatisfaction']
    
    X_test_orig = test_sets[i][['Log Income', 'Educational Achievement', 'Employed', 
                                'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                                'Emotional Health']]
    
    Train_test_linreg_orig = linreg_train_test(X_train = X_train_orig, 
                                               y_train = y_train_orig, 
                                               X_test = X_test_orig, 
                                               y_test = y_test_orig)
    
    MSEs_test_orig.append(Train_test_linreg_orig[0])
    
    MSEs_train_orig.append(Train_test_linreg_orig[1])
    
results_linregs_orig = pd.DataFrame([MSEs_test_orig, MSEs_train_orig])    

results_linregs_orig_T = results_linregs_orig.T

results_linregs_orig_T.loc['Means'] = results_linregs_orig_T.mean()

results_linregs_orig_T.loc['SDs'] = results_linregs_orig_T.std()

results_linregs_orig_T.columns = ['Test MSEs', 'Train MSEs']    

#################################
####JOINED MODEL WITH OHE, 72####   
#################################

#As specified in the paper, in running the Extended set, we drop the originals
#'Employed', 'Educational Achievement' and 'Has a Partner', as already implied
#by other variables.
#We also drop the dummy "Any long-standing illness", for the same reason.

MSEs_test_joined = []

MSEs_train_joined = []
    
for i in range(0, len(training_sets)):
    
    y_train_joined = training_sets[i]['lifesatisfaction']
    
    X_train_joined = training_sets[i].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Any long-standing illness'], axis = 1)
    
    y_test_joined = test_sets[i]['lifesatisfaction']
    
    X_test_joined = test_sets[i].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Any long-standing illness'], axis = 1)
          
    Train_test_linreg_joined = linreg_train_test(X_train = X_train_joined, 
                                                 y_train = y_train_joined, 
                                                 X_test = X_test_joined, 
                                                 y_test = y_test_joined)
    
    MSEs_test_joined.append(Train_test_linreg_joined[0])
    
    MSEs_train_joined.append(Train_test_linreg_joined[1])
    
results_linregs_joined = pd.DataFrame([MSEs_test_joined, MSEs_train_joined])    

results_linregs_joined_T = results_linregs_joined.T

results_linregs_joined_T.loc['Means'] = results_linregs_joined_T.mean()

results_linregs_joined_T.loc['SDs'] = results_linregs_joined_T.std()

results_linregs_joined_T.columns = ['Test MSEs', 'Train MSEs']   
