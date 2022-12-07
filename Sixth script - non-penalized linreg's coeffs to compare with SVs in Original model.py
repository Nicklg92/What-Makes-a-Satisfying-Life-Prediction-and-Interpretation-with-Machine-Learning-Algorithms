#########################################################################
###SIXTH SCRIPT - LINREG'S COEFFICIENTS TO COMPARE WITH SHAPLEY VALUES###
#########################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
'''
COMMENTS:

This is the sixth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).
    
Aim of these next three scripts is to produce the values in Table 9 and
10 of the paper, comparing the Linear Regression's Coefficients with the
Random Forest's Shapley Values.

In this one, we simply extract the coefficients for the Non-Penalized
Linear Regression used for the Original model.  

Without loss of generality, we derived the coefficients only for the
train-test split 1. The coefficients are computed on the training-set
insample regression. No testing is required here.
'''

path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_Test_splits'  

dest_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS'

#We create a function to fot a linear regression only in the training set
#and extract the coefficients.

def linreg_insample(X, y):
    
    X_const = sm.add_constant(X)

    model_const = sm.OLS(y, X_const).fit()

    print(model_const.summary())
    
    res = model_const.resid 
    
    predicted = model_const.predict()
    
    list_of_values = [model_const, res, predicted]
        
    return list_of_values

train_1 = pd.read_csv(path + '\\train_1.csv')

#The other operations are the same we did when running the linregs
#to predict on the test set.

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

train_1['Physical Health'] = train_1['Physical Health'].apply(lambda x: -1 * x)

##################################
##COEFFICIENTS ORIGINAL, SPLIT 1##
##################################

y_train_orig_1 = train_1['lifesatisfaction']
    
X_train_orig_1 = train_1[['Log Income', 'Educational Achievement', 'Employed', 
                        'Good Conduct', 'Female', 'Has a Partner', 'Physical Health', 
                        'Emotional Health']]

    
Insample_training_orig_1 = linreg_insample(X_train_orig_1, y_train_orig_1)

Insample_training_orig_1_summary = Insample_training_orig_1[0].summary()

Insample_training_orig_1_summary_as_html = Insample_training_orig_1_summary.tables[1].as_html()

Insample_training_orig_1_summary_as_pd = pd.read_html(Insample_training_orig_1_summary_as_html, header = 0, index_col = 0)[0]

#We manually put as 0 all the nonsignificant coefficients at 95% level.

Insample_training_orig_1_summary_as_pd.loc[Insample_training_orig_1_summary_as_pd['P>|t|'] > 0.05, 'coef'] = 0

Coeffs_training_orig_1 = Insample_training_orig_1_summary_as_pd['coef']

Coeffs_training_orig_1.to_csv(dest_path + '\\Train_1_orig_linreg_coeffs.csv')
