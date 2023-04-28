###################################
###FIFTH SCRIPT - RANDOM FORESTS###
###################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eli5
import shap
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

np.random.seed(42)

'''
COMMENTS:
    
This is the fifth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

Here, we fit and predict using Random Forests on all the 100 train-test
splits, in particular those with categorical variables not one-hot-encoded.

For both the Original and Extended datasets, we find the optimal Random Forest 
(with 4000 trees) and, on it, compute and plot the individual Shapley 
Values (colored bar chart).

Then, using those same parameters, we run the 100 Random Forests, and 
on those compute the Average Test MSEs, Average Train MSEs, and Average
Mean Absolute Shaley Values, as shown in the paper.

'''    

#We create a custom function to fit and predict with random forest.
#We remind the reader to the paper for a description of what the 
#hyperparameters represent.

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

dest_path = 'C:\\some_local_path\\'

#Importing training sets

training_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + 'train_noohed_' + j + '.csv'

    training_sets.append(pd.read_csv(import_path))

#Importing test set

test_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + 'test_noohed_' + j + '.csv'
        
    test_sets.append(pd.read_csv(import_path))
    
#We invert the values of physical health to represent 
#good rather than bad health.

for i in range(0, len(training_sets)):
    
    training_sets[i]['Physical Health'] = training_sets[i]['Physical Health'].apply(lambda x: -1 * x)
    
    test_sets[i]['Physical Health'] = test_sets[i]['Physical Health'].apply(lambda x: -1 * x)
    
#########################
###ORIGINAL 8 FEATURES###
#########################

#As discussed in the paper, we consider the first train-test split (without loss
#of generality) to learn the hyperparameters of the Random Forest to then
#be used on all the train-test splits.

#On this same split we also produce the Shapley Values at the individual
#level, as in Figure 2.

y_train_orig_1 = training_sets[0]['lifesatisfaction']
    
X_train_orig_1 = training_sets[0][['Log Income', 'Educational Achievement', 'Employed', 
                                   'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                                   'Emotional Health']]
    
y_test_orig_1 = test_sets[0]['lifesatisfaction']
    
X_test_orig_1 = test_sets[0][['Log Income', 'Educational Achievement', 'Employed', 
                            'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                            'Emotional Health']]

#We consider a large set of hyperparameters.
#The finally presented grids are derie-ved also from other experiments,
#including smaller and larger values of those presented here.

#The value of the hyperparameters balance off bias and variance of the
#resulting algorithm.

#The final values for the optimal hyperparameters are those presented
#in the paper.

#Depending on the available resources, tha below can take up to
#1 hour to run.

Random_Forest_orig = RandomForest(X_train = X_train_orig_1, 
                                  y_train = pd.DataFrame(y_train_orig_1),
                                  if_bootstrap = True,
                                  n_max_feats = [1, 'sqrt', 'auto'],
                                  n_trees = [4000],
                                  n_max_depth = [5,6,7,8,9,10,11,12,13],
                                  n_min_sample_leaf = [8,9,10,11,12,13,14,15,16],
                                  n_cv = 5,
                                  X_test = X_test_orig_1, 
                                  y_test = pd.DataFrame(y_test_orig_1)) 

Random_Forest_orig[-3]

MSE_Train_RF_original = Random_Forest_orig[-1]

MSE_Test_RF_original = Random_Forest_orig[-2]

########################################
##SHAPLEY VALUES ON TRAIN-TEST SPLIT 1##
########################################

#Now, we will ex-ante fix the hyperparameters found previously, and on those
#we will compute the Shapley Values at an individual level for train 
#set 1.

explainer_orig = shap.TreeExplainer(Random_Forest_orig[-3])

shap_values_orig = explainer_orig.shap_values(X_train_orig_1)

shap.summary_plot(shap_values_orig, X_train_orig_1)

shap_values_df_orig = pd.DataFrame(shap_values_orig)

shap_values_df_orig.columns = list(X_train_orig_1)

shap_values_df_orig.to_csv(dest_path + 'Shapley_Values_train_1_orig_4000_trees.csv')

########################################################
###SHAPLEY VALUES AND MSEs: THE 100 TRAIN-TEST SPLITS###
########################################################

#We now use the previously found hyperparameters on train-test split 1
#to fit and predict using random forests on all the other 100 splits.

#On each of them, moreover, we compute the Mean Absolue Shapley Values for
#each split - equation 13 in the paper, and further average them across all
#the splits - producing the Average Mean Absolute Shapley Values - 
#equation 14 in the paper, and associated Figure 1 and Table 7.

MSEs_test_orig = []

MSEs_train_orig = []

shap_values = []

for i in range(0, len(training_sets)):
    
    y_train_orig_i = training_sets[i]['lifesatisfaction']
    
    X_train_orig_i = training_sets[i][['Log Income', 'Educational Achievement', 'Employed', 
                                       'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                                       'Emotional Health']]
    
    y_test_orig_i = test_sets[i]['lifesatisfaction']
    
    X_test_orig_i = test_sets[i][['Log Income', 'Educational Achievement', 'Employed', 
                                  'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                                  'Emotional Health']]   
    
    #Notice that since the hyperparameters are ex-ante fixed, cross-validation
    #is no longer needed. We leave the minimum possible value 2 just to
    #avoid getting an error.
    
    #In the for loop, we also already compute the Shapley Values and store
    #them in dataframes, to then be used to produce the graphics.
    
    Random_Forest_orig_i = RandomForest(X_train = X_train_orig_i,
                                        y_train = pd.DataFrame(y_train_orig_i),
                                        if_bootstrap = True,
                                        n_max_feats = ['sqrt'],
                                        n_trees = [400],
                                        n_max_depth = [8],
                                        n_min_sample_leaf = [15],
                                        n_cv = 2,
                                        X_test = X_test_orig_i, 
                                        y_test = pd.DataFrame(y_test_orig_i)) 
    
    MSEs_test_orig.append(Random_Forest_orig_i[-2])
    
    MSEs_train_orig.append(Random_Forest_orig_i[-1])
    
    explainer_orig_i = shap.TreeExplainer(Random_Forest_orig_i[-3])

    shap_values_orig_i = explainer_orig_i.shap_values(X_train_orig_i)
    
    shap_values.append(pd.DataFrame(shap_values_orig_i))
    
abs_mean_shaps_each_trset = []
    
for i in range(len(shap_values)):
    
    abs_mean_shaps_each_trset.append(shap_values[i].abs().mean())
    
abs_mean_shapleys = pd.concat(abs_mean_shaps_each_trset, axis = 1) 

abs_mean_shapleys_T = abs_mean_shapleys.T

#All the 100 training sets have the same features' names, clearly.

shap.summary_plot(abs_mean_shapleys_T, X_train_orig_i, plot_type = "bar")

#And to do it manually (Mean Absolute Shapley Value = MASV)

abs_mean_shapleys_T.loc['Mean MASV'] = abs_mean_shapleys_T.mean()

abs_mean_shapleys_T.loc['Std MASV'] = abs_mean_shapleys_T.std()

abs_mean_shapleys_T.columns = list(X_train_orig_i)

MSEs_test_pd = pd.DataFrame(MSEs_test_orig)

MSEs_test_pd.loc['Mean Test MSE'] = MSEs_test_pd.mean()

MSEs_test_pd.loc['Std Test MSE'] = MSEs_test_pd.std()

MSEs_train_pd = pd.DataFrame(MSEs_train_orig)

MSEs_train_pd.loc['Mean Train MSE'] = MSEs_train_pd.mean()

MSEs_train_pd.loc['Std Train MSE'] = MSEs_train_pd.std()

abs_mean_shapleys_T.to_csv(dest_path + 'Average_MASVs_orig.csv')

MSEs_train_pd.to_csv(dest_path + 'MSEs_train_100_orig.csv')

MSEs_test_pd.to_csv(dest_path + 'MSEs_test_100_orig.csv') 

##########################################################
###PERMUTATION IMPORTANCE ORIGINAL - TRAIN-TEST SPLIT 1###
##########################################################

#We conclude computing the Permutation Importance on the
#first train - test split (in particular, the test set)
#as reported in Table 11.

PI_orig = PermutationImportance(Random_Forest_orig[-3], n_iter = 100)

PI_test_set_orig = PI_orig.fit(X = X_test_orig_1,
                               y = y_test_orig_1)

PI_test_set_orig.feature_importances_

np.sum(PI_test_set_orig.feature_importances_ <= 0)

eli5.show_weights(PI_test_set_orig, 
                  top = 9, 
                  feature_names = X_test_orig_1.columns.tolist()).data

#Adding the standard deviations.

pi_features = eli5.explain_weights_df(PI_test_set_orig, feature_names = X_test_orig_1.columns.tolist())    

pi_features.to_csv(dest_path + 'PI_orig_test_1_4000_trees.csv')

#################################################################
#################################################################
#################################################################

#We redo all of the above for the Extended Set specularly.

##############################
###EXTENDED SET OF FEATURES###
##############################

#In this case, we made some inversions also on Tenure Status and Marital
#Status, for readability in Shapley Values' graph.

#Nothing changes in terms of results.

#Please note that in this case running all the computations on the 
#Extended set may require multiple hours (in parallel!).

for i in range(0, len(training_sets)):
    
    training_sets[i]['Marital Status'].replace([1,2,3,4,5,6,-7], [6,5,4,3,2,1,-1], inplace = True)
    
    test_sets[i]['Marital Status'].replace([1,2,3,4,5,6,-7], [6,5,4,3,2,1,-1], inplace = True)
    
    training_sets[i]['Tenure Status'].replace([1,2,3,4,5,6,7,8,-8,-9], [8,7,6,5,4,3,2,1,-1,-2], inplace = True)
    
    test_sets[i]['Tenure Status'].replace([1,2,3,4,5,6,7,8,-8,-9], [8,7,6,5,4,3,2,1,-1,-2], inplace = True)
    
y_train_joined_1 = training_sets[0]['lifesatisfaction']
    
X_train_joined_1 = training_sets[0].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Unnamed: 0'], axis = 1)
    
y_test_joined_1 = test_sets[0]['lifesatisfaction']
    
X_test_joined_1 = test_sets[0].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Unnamed: 0'], axis = 1)

Random_Forest_joined = RandomForest(X_train = X_train_joined_1, 
                                    y_train = pd.DataFrame(y_train_joined_1),
                                    if_bootstrap = True,
                                    n_max_feats = [1, 'sqrt', 'auto'],
                                    n_trees = [4000],
                                    n_max_depth = [4,5,6,7,8,9,10,11,12,13,14,15,16],
                                    n_min_sample_leaf = [1,2,3,4,5,6,7,8,9,10,11,12],
                                    n_cv = 5,
                                    X_test = X_test_joined_1, 
                                    y_test = pd.DataFrame(y_test_joined_1)) 

Random_Forest_joined[-3]

MSE_Train_RF_joined = Random_Forest_joined[-1]

MSE_Test_RF_joined = Random_Forest_joined[-2]

########################################
##SHAPLEY VALUES ON TRAIN-TEST SPLIT 1##
########################################

explainer_joined = shap.TreeExplainer(Random_Forest_joined[-3])

shap_values_joined = explainer_joined.shap_values(X_train_joined_1)

shap.summary_plot(shap_values_joined, X_train_joined_1)

shap_values_df_joined = pd.DataFrame(shap_values_joined)

shap_values_df_joined.columns = list(X_train_joined_1)

shap_values_df_joined.to_csv(dest_path + 'Shapley_Values_train_1_joined_4000_trees.csv')

###################################################
###SHAPLEY VALUES AND MSEs: CONFIDENCE INTERVALS###
###################################################

MSEs_test_joined = []

MSEs_train_joined = []

shap_values = []

for i in range(0, len(training_sets)):
    
    y_train_joined_i = training_sets[i]['lifesatisfaction']
    
    X_train_joined_i = training_sets[i].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Unnamed: 0'], axis = 1)
    
    y_test_joined_i = test_sets[i]['lifesatisfaction']
    
    X_test_joined_i = test_sets[i].drop(['lifesatisfaction', 'Employed', 'Educational Achievement', 'Has a Partner', 'Unnamed: 0'], axis = 1)  
    
    Random_Forest_joined_i = RandomForest(X_train = X_train_joined_i,
                                          y_train = pd.DataFrame(y_train_joined_i),
                                          if_bootstrap = True,
                                          n_max_feats = ['sqrt'],
                                          n_trees = [400],
                                          n_max_depth = [13],
                                          n_min_sample_leaf = [8],
                                          n_cv = 2,
                                          X_test = X_test_joined_i, 
                                          y_test = pd.DataFrame(y_test_joined_i)) 
    
    MSEs_test_joined.append(Random_Forest_joined_i[-2])
    
    MSEs_train_joined.append(Random_Forest_joined_i[-1])
    
    explainer_joined_i = shap.TreeExplainer(Random_Forest_joined_i[-3])

    shap_values_joined_i = explainer_joined_i.shap_values(X_train_joined_i)
    
    shap_values.append(pd.DataFrame(shap_values_joined_i))
    
abs_mean_shaps_each_trset = []
    
for i in range(len(shap_values)):
    
    abs_mean_shaps_each_trset.append(shap_values[i].abs().mean())
    
abs_mean_shapleys = pd.concat(abs_mean_shaps_each_trset, axis = 1) 

abs_mean_shapleys_T = abs_mean_shapleys.T

shap.summary_plot(abs_mean_shapleys_T, X_train_joined_i, plot_type = "bar")

abs_mean_shapleys_T.loc['Mean MASV'] = abs_mean_shapleys_T.mean()

abs_mean_shapleys_T.loc['Std MASV'] = abs_mean_shapleys_T.std()

abs_mean_shapleys_T.columns = list(X_train_joined_i)

MSEs_test_pd = pd.DataFrame(MSEs_test_joined)

MSEs_test_pd.loc['Mean Test MSE'] = MSEs_test_pd.mean()

MSEs_test_pd.loc['Std Test MSE'] = MSEs_test_pd.std()

MSEs_train_pd = pd.DataFrame(MSEs_train_joined)

MSEs_train_pd.loc['Mean Train MSE'] = MSEs_train_pd.mean()

MSEs_train_pd.loc['Std Train MSE'] = MSEs_train_pd.std()

abs_mean_shapleys_T.to_csv(dest_path + 'Average_MASVs_joined.csv')

MSEs_train_pd.to_csv(dest_path + 'MSEs_train_100_joined.csv')

MSEs_test_pd.to_csv(dest_path + 'MSEs_test_100_joined.csv')

###################################
###PERMUTATION IMPORTANCE JOINED###
###################################

PI_joined = PermutationImportance(Random_Forest_joined[-3], n_iter=100)

PI_test_set_joined = PI_joined.fit(X = X_test_joined_1,
                                   y = y_test_joined_1)


PI_test_set_joined.feature_importances_

np.sum(PI_test_set_joined.feature_importances_ <= 0)

eli5.show_weights(PI_test_set_joined, 
                  top = 23, 
                  feature_names = X_test_joined_1.columns.tolist()).data

#Adding the standard deviations.

pi_features = eli5.explain_weights_df(PI_test_set_joined, feature_names = X_test_joined_1.columns.tolist())    

pi_features.to_csv(dest_path + 'PI_joined_test_1_4000_trees.csv')
