####################################################################
###TWELFTH SCRIPT - RANDOM FOREST LEARNING CURVES, EXTENDED MODEL###
####################################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

'''
COMMENTS:
    
This is the twelfth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

In this script, in particular, we create the data for Figure 8, the Learning
Curve of the Random Forest on the Extended model.

Same logic as for the Learning Curve of the Random Forest of the Original model.
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

path = 'C:\\some_local_path_noohed\\'

dest_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\'

train_1 = pd.read_csv(path + 'train_noohed_1.csv')

test_1 = pd.read_csv(path + 'test_noohed_1.csv')

for i in [train_1, test_1]:
        
    i.drop(['Employed', 'Educational Achievement', 'Has a Partner', 'Any long-standing illness', 'Unnamed: 0'], axis = 1, inplace = True)
        
    i['Physical Health'] = i['Physical Health'].apply(lambda x: -1 * x)
    
    i['Marital Status'].replace([1,2,3,4,5,6,-7], [6,5,4,3,2,1,-1], inplace = True)

    i['Tenure Status'].replace([1,2,3,4,5,6,7,8,-8,-9], [8,7,6,5,4,3,2,1,-1,-2], inplace = True)
    
X_train_1 = train_1.drop(['lifesatisfaction'], axis = 1)  

X_test_1 = test_1.drop(['lifesatisfaction'], axis = 1)  

y_train_1 = train_1['lifesatisfaction']

y_test_1 = test_1['lifesatisfaction']

X = pd.concat([X_train_1, X_test_1], ignore_index = True)

y = pd.concat([y_train_1, y_test_1], ignore_index = True)

####################
###EXTENDED MODEL###
####################

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size = 0.99,
                                                    random_state = 42)

Random_Forest_joined = RandomForest(X_train = X_train, 
                                    y_train = pd.DataFrame(y_train),
                                    if_bootstrap = True,
                                    n_max_feats = [1, 'sqrt', 'auto'],
                                    n_trees = [400],
                                    n_max_depth = [1,2,3,4,5,6],
                                    n_min_sample_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                                    n_cv = 5,
                                    X_test = X_test, 
                                    y_test = pd.DataFrame(y_test)) 

MSE_Train_joined = Random_Forest_joined[-1]

MSE_Test_joined = Random_Forest_joined[-2]

Best_RF = Random_Forest_joined[-3]

####################
##80 perc Training##
####################

#MSE_Train_joined = 2.25
#MSE_Test_joined = 2.68

#sqrt ex-qnte
#400 trees ex-ante
#max depth = 10 in [10,11,12,13,14,15,16]
#min sample leaf = 7 in [4,5,6,7,8,9,10]


####################
##70 perc Training##
####################

#MSE_Train_joined = 2.24
#MSE_Test_joined = 2.66

#sqrt ex-qnte
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11]
#min sample leaf = 7 in [4,5,6,7,8,9,10]


####################
##60 perc Training##
####################

#MSE_Train_joined = 2.19
#MSE_Test_joined = 2.72

#sqrt ex-qnte
#400 trees ex-ante
#max depth = 9 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 5 in [4,5,6,7,8,9,10]


####################
##50 perc Training##
####################

#MSE_Train_joined = 2.14
#MSE_Test_joined = 2.74

#sqrt ex-ante
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 7 in [4,5,6,7,8,9,10]


####################
##40 perc Training##
####################

#MSE_Train_joined = 2.12
#MSE_Test_joined = 2.73

#sqrt ex-qnte
#400 trees ex-ante
#max depth = 11 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 8 in [4,5,6,7,8,9,10]


####################
##30 perc Training##
####################

#MSE_Train_joined = 2.06
#MSE_Test_joined = 2.70

#sqrt ex-qnte
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 6 in [4,5,6,7,8,9,10]

####################
##20 perc Training##
####################

#MSE_Train_joined = 1.81
#MSE_Test_joined = 2.73

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 11 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 5 in [4,5,6,7,8,9,10]

####################
##10 perc Training##
####################

#MSE_Train_joined = 1.61
#MSE_Test_joined = 2.79

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 12 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 5 in [4,5,6,7,8,9,10]


###################
##9 perc Training##
###################

#MSE_Train_joined = 1.66
#MSE_Test_joined = 2.78

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 5 in [4,5,6,7,8,9,10]


###################
##8 perc Training##
###################

#MSE_Train_joined = 1.48
#MSE_Test_joined = 2.78

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 11 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 4 in [4,5,6,7,8,9,10]

###################
##7 perc Training##
###################

#MSE_Train_joined = 0.83
#MSE_Test_joined = 2.81

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 1 in [1,2,3,4,5,6,7,8,9]

###################
##6 perc Training##
###################

#MSE_Train_joined = 0.78
#MSE_Test_joined = 2.83

#1 in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 12 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 1 in [1,2,3,4,5,6,7,8,9]

###################
##5 perc Training##
###################

#MSE_Train_joined = 1.62
#MSE_Test_joined = 2.81

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 9 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 4 in [1,2,3,4,5,6,7,8,9]


###################
##4 perc Training##
###################

#MSE_Train_joined = 1.61
#MSE_Test_joined = 2.81

#sqrt in [1, 'sqrt', 'auto']
#400 trees ex-ante
#max depth = 10 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 4 in [1,2,3,4,5,6,7,8,9] 

###################
##3 perc Training##
###################

#MSE_Train_joined = 2.23
#MSE_Test_joined = 2.83

#sqrt in [1, 'sqrt', 'auto'] 
#400 trees ex-ante
#max depth = 8 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 9 in [1,2,3,4,5,6,7,8,9] 

###################
##2 perc Training##
###################

#MSE_Train_joined = 1.39
#MSE_Test_joined = 2.86

#sqrt in [1, 'sqrt', 'auto'] 
#400 trees ex-ante
#max depth = 11 in [5,6,7,8,9,10,11,12,13]
#min sample leaf = 4 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14] 

###################
##1 perc Training##
###################

#MSE_Train_joined = 1.22
#MSE_Test_joined = 2.85

#'sqrt' in [1, 'sqrt', 'auto'] 
#400 trees ex-ante
#max depth = 5 in [1,2,3,4,5,6]
#min sample leaf = 3 in [1,2,3,4,5,6,7,8,9,10,11,12,13,14] 

