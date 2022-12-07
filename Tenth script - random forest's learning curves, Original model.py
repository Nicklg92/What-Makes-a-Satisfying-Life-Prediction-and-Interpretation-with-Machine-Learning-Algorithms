##################################################################
###TENTH SCRIPT - RANDOM FOREST LEARNING CURVES, ORIGINAL MODEL###
##################################################################

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

'''
COMMENTS:
    
This is the tenth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

In this script, in particular, we create the data for Figure 7, the Learning
Curve of the Random Forest on the Original model.

Differently from the Linear Regression, however, in this case the study
of each training size was done separately. The reason bing that associated to each 
training size there were very different hyperparametric grids to be considered.  

A part from that, the same considerations for the Learning Curves on the
Non-Penalized Linear Regression apply.
'''

def RandomForest(X_train, y_train, if_bootstrap,
                 n_trees, n_max_feats, 
                 n_max_depth, n_min_sample_leaf, 
                 n_cv, X_test, y_test):
        
    rf = RandomForestRegressor(bootstrap = if_bootstrap)

    pruning_dict = {'n_estimators':n_trees,
                    'max_features': n_max_feats,
                    'max_depth':n_max_depth,
                    'min_samples_leaf':n_min_sample_leaf}
        
    rf_regr_optim = GridSearchCV(rf, pruning_dict, 
                                 cv = n_cv, n_jobs = -1, 
                                 scoring='neg_mean_squared_error')
        
    rf_regr_fitted = rf_regr_optim.fit(X_train, y_train)
        
    best_rf = rf_regr_fitted.best_estimator_
    
    results_from_cv = rf_regr_fitted.cv_results_
        
    rf_regr_yhat = rf_regr_fitted.predict(X_test)
        
    rf_regr_yhat_to_use = rf_regr_yhat.reshape((len(y_test), 1))

    MSE_test = ((rf_regr_yhat_to_use - y_test)**2).mean()
        
    rf_regr_yhat_train = rf_regr_fitted.predict(X_train)
        
    rf_regr_yhat_to_use_train = rf_regr_yhat_train.reshape((len(y_train), 1))

    MSE_train = ((rf_regr_yhat_to_use_train - y_train)**2).mean()
    
    list_of_results = [rf_regr_fitted, results_from_cv, best_rf, MSE_test, MSE_train]
    
    return list_of_results

path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits_noohed\\'

dest_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\'

train_1 = pd.read_csv(path + 'train_noohed_1.csv')

test_1 = pd.read_csv(path + 'test_noohed_1.csv')

for i in [train_1, test_1]:
    
    i.drop(['Any long-standing illness', 'Unnamed: 0'], axis = 1, inplace = True)
        
    i['Physical Health'] = i['Physical Health'].apply(lambda x: -1 * x)

X_train_1 = train_1.drop(['lifesatisfaction'], axis = 1)  

X_test_1 = test_1.drop(['lifesatisfaction'], axis = 1)  

y_train_1 = train_1['lifesatisfaction']

y_test_1 = test_1['lifesatisfaction']

X = pd.concat([X_train_1, X_test_1], ignore_index = True)

y = pd.concat([y_train_1, y_test_1], ignore_index = True)

#In the following, the Random Forest was optimized for each training size.
#The pattern of results of the optimization process are described each time.
#The optimizations were done on forests with 400 trees, yielding comparable
#results to ones with 4000.

####################
###ORIGINAL MODEL###
####################

X_orig = X[['Log Income', 'Educational Achievement', 'Employed', 
            'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
            'Emotional Health']]

X_train, X_test, y_train, y_test = train_test_split(X_orig, 
                                                    y,
                                                    test_size = 0.99,
                                                    random_state = 42)

Random_Forest_orig = RandomForest(X_train = X_train, 
                                  y_train = pd.DataFrame(y_train),
                                  if_bootstrap = True,
                                  n_max_feats = [1, 'sqrt', 'auto'],
                                  n_trees = [400],
                                  n_max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                                  n_min_sample_leaf = [8,9,10,11,12,13,14,15,16,17,18,19,20,21],
                                  n_cv = 5,
                                  X_test = X_test, 
                                  y_test = pd.DataFrame(y_test)) 


Random_Forest_orig[-3]

MSE_Train_RF_original = Random_Forest_orig[-1]

MSE_Test_RF_original = Random_Forest_orig[-2]

##################
###80% TRAINING###
##################

#MSE_Train_RF_original = 2.66
#MSE_Test_RF_original = 2.73

#400 trees ex-ante
#sqrt ex-ante
#max depth = 9 in [5,6,7,8,9,10,11,12,13]
#min_samples_leaf = 19 in [15,16,17,18,19,20,21,22]

##################
###70% TRAINING###
##################

#MSE_Train_RF_original = 2.63
#MSE_Test_RF_original = 2.76

#400 trees ex-ante
#sqrt ex-ante
#max depth = 9 in [5,6,7,8,9,10,11,12,13]
#min_samples_leaf = 14 in [11,12,13,14,15,16,17,18,19,20]

##################
###60% TRAINING###
##################

#MSE_Train_RF_original = 2.58
#MSE_Test_RF_original = 2.83

#400 trees ex-ante
#sqrt ex-ante
#max depth = 10 in [4,5,6,7,8,9,10,11,12,13,14]
#min_samples_leaf = 15 in [7,8,9,10,11,12,13,14,15,16,17,18,19]

##################
###50% TRAINING###
##################


#MSE_Train_RF_original = 2.45
#MSE_Test_RF_original = 2.87

#400 trees ex-ante
#sqrt ex-ante
#max depth = 11 in [4,5,6,7,8,9,10,11,12,13,14]
#min_samples_leaf = 9 in [7,8,9,10,11,12,13,14,15,16,17,18,19]

##################
###40% TRAINING###
##################

#MSE_Train_RF_original = 2.49
#MSE_Test_RF_original = 2.84

#400 trees ex-ante
#sqrt ex-ante
#max depth = 9 in [4,5,6,7,8,9,10,11,12,13,14]
#min_samples_leaf = 8 in [7,8,9,10,11,12,13,14,15,16,17,18,19]

##################
###30% TRAINING###
##################

#MSE_Train_RF_original = 2.50
#MSE_Test_RF_original = 2.82

#400 trees ex-ante
#sqrt in [1, 'sqrt', 'auto']
#max depth = 8 in [4,5,6,7,8,9,10,11,12]
#min_samples_leaf = 7 in [4,5,6,7,8,9,10,11,12,13,14,15,16]

##################
###20% TRAINING###
##################

#MSE_Train_RF_original = 2.24
#MSE_Test_RF_original = 2.86

#400 trees ex-ante
#sqrt in [1, 'sqrt', 'auto']
#max depth = 9 in [4,5,6,7,8,9,10,11,12]
#min_samples_leaf = 4 in [4,5,6,7,8,9,10,11,12,13,14,15,16]

##################
###10% TRAINING###
##################

#MSE_Train_RF_original = 2.15
#MSE_Test_RF_original = 2.88
 
#400 trees ex-ante
#'sqrt' in [1, 'sqrt', 'auto']
#max depth = 7 in [2,3,4,5,6,7,8]
#min_samples_leaf = 5 in [1,2,3,4,5,6,7,8,9,10,11,12,13]

#################
###9% TRAINING###
##################

#MSE_Train_RF_original = 2.33
#MSE_Test_RF_original = 2.88

#400 trees ex-ante
#'sqrt' in [1, 'sqrt', 'auto']
#max depth = 12 in [4,5,6,7,8,9,10,11,12]
#min_samples_leaf = 15 in [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#################
###8% TRAINING###
#################

#MSE_Train_RF_original = 2.25
#MSE_Test_RF_original = 2.88

#400 trees ex-ante
#'sqrt' in [1, 'sqrt', 'auto']
#max depth = 8 in [8,9,10,11,12,13,14,15,16]
#min_samples_leaf = 10 in [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#################
###7% TRAINING###
#################

#MSE_Train_RF_original = 2.05
#MSE_Test_RF_original = 2.89

#400 trees ex-ante
#'sqrt' in [1, 'sqrt', 'auto']
#max depth = 7 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 4 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#################
###6% TRAINING###
##################

#MSE_Train_RF_original = 2.39
#MSE_Test_RF_original = 2.91

#400 trees ex-ante
#'auto' in [1, 'sqrt', 'auto']
#max depth = 5 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 17 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

#################
###5% TRAINING###
#################

#MSE_Train_RF_original = 2.50
#MSE_Test_RF_original = 2.90

#400 trees ex-ante
#'sqrt' in [1, 'sqrt', 'auto']
#max depth = 13 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 14 in [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

#################
###4% TRAINING###
#################

#MSE_Train_RF_original = 2.52
#MSE_Test_RF_original = 2.89

#400 trees ex-ante
#'auto' in [1, 'sqrt', 'auto']
#max depth = 4 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 21 in [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

#################
###3% TRAINING###
#################

#MSE_Train_RF_original = 2.58
#MSE_Test_RF_original = 2.92

#400 trees ex-ante
#'auto' in [1, 'sqrt', 'auto']
#max depth = 10 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 17 in [16,17,18,19,20,21,22,23,24,25,26]

#################
###2% TRAINING###
#################

#MSE_Train_RF_original = 2.21
#MSE_Test_RF_original = 3.07

#400 trees ex-ante
#'auto' in [1, 'sqrt', 'auto']
#max depth = 2 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 13 in [12,13,14,15,16,17,18,19,20,21,22,23]

#################
###1% TRAINING###
#################

#MSE_Train_RF_original = 2.20
#MSE_Test_RF_original = 3.00

#400 trees ex-ante
#'auto' in [1, 'sqrt', 'auto']
#max depth = 6 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#min_samples_leaf = 13 in [8,9,10,11,12,13,14,15,16,17,18,19,20,21]