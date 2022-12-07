#################################################################
###THIRTEENTH'S SCRIPT - PRODUCING THE LEARNING CURVES' GRAPHS###
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
COMMENTS:
    
This is the thirtheenth and final script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

Here, we simply wrap-up all the data on the Learning Curves produced in the 
previous four scripts and create the plots.    
'''

import_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\'

#As can be seen in the four scripts, for the Non-Penalized Linear and Ridge
#Regressions then Test MSEs were automatically saved in csv files. 
#Conversely, this was not the case for the Random Forests, hence the 
#values are manually reported here.

#Importing Learning Curve Non-Penalized Linear Regression, Original model.

linreg_orig_lc = pd.read_csv(import_path + 'Linreg_orig_learning_curve.csv')

linreg_orig_lc.drop(['Unnamed: 0'], axis = 1, inplace = True)

linreg_orig_lc.columns = ['Test MSEs', 'Train sizes']

linreg_orig_lc['Train sizes'] = linreg_orig_lc['Train sizes'].apply(lambda x: x / 100)

#Importing Learning Curve Ridge Regression, Extended model.

ridgereg_extd_lc = pd.read_csv(import_path + 'Ridge_extd_learning_curve.csv')

ridgereg_extd_lc.drop(['Unnamed: 0'], axis = 1, inplace = True)

ridgereg_extd_lc.columns = ['Test MSEs', 'Train sizes']

ridgereg_extd_lc['Train sizes'] = ridgereg_extd_lc['Train sizes'].apply(lambda x: x / 100)

#Creating Learning Curve for Random Forest, Original model. Vaulues are copy-pasted
#from the results section of the associated script.

train_sizes = [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6,
               5, 4, 3, 2, 1] 

rf_orig_lc_test = [2.73, 2.76, 2.83, 2.87, 2.84, 2.82, 2.86, 2.88, 2.88, 2.88, 2.89, 2.91, 2.90, 2.89, 2.92, 3.07, 3.00]

rf_orig_lc = pd.DataFrame([rf_orig_lc_test, train_sizes]).T

rf_orig_lc.columns = ['Test MSEs', 'Train sizes']

rf_orig_lc['Train sizes'] = rf_orig_lc['Train sizes'].apply(lambda x: x / 100)

#Creating Learning Curve for Random Forest, Extended model. Vaulues are copy-pasted
#from the results section of the associated script.

rf_extd_lc_test = [2.68, 2.66, 2.72, 2.74, 2.73, 2.70, 2.73, 2.79, 2.78, 2.78, 2.81, 2.83, 2.81, 2.81, 2.83, 2.86, 2.85]

rf_extd_lc = pd.DataFrame([rf_extd_lc_test, train_sizes]).T

rf_extd_lc.columns = ['Test MSEs', 'Train sizes']

rf_extd_lc['Train sizes'] = rf_extd_lc['Train sizes'].apply(lambda x: x / 100)

######################
###LINREG ORIG PLOT###
######################

fig, ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(22)
ax.plot(linreg_orig_lc['Train sizes'], linreg_orig_lc['Test MSEs'], color = 'b', marker = 'o', linestyle = '--')
ax.set_xlabel('Train size')
ax.set_ylabel('MSE')
ax.set_title('Test MSEs per training size Linear Regression, Original Model')
plt.xticks(list(linreg_orig_lc['Train sizes']), rotation = 90)
plt.legend(loc = "upper right")
plt.show()


########################
###RIDGEREG EXTD PLOT###
########################

fig, ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(22)
ax.plot(ridgereg_extd_lc['Train sizes'], ridgereg_extd_lc['Test MSEs'], color = 'b', marker = 'o', linestyle = '--')
ax.set_xlabel('Train size')
ax.set_ylabel('MSE')
ax.set_title('Test MSEs per training size Ridge Regression, Extended Model')
plt.xticks(list(ridgereg_extd_lc['Train sizes']), rotation = 90)
plt.legend(loc = "upper right")
plt.show()


##################
###RF ORIG PLOT###
##################

fig, ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(22)
ax.plot(rf_orig_lc['Train sizes'], rf_orig_lc['Test MSEs'], color = 'b', marker = 'o', linestyle = '--')
ax.set_xlabel('Train size')
ax.set_ylabel('MSE')
ax.set_title('Test MSEs per training size Random Forest, Original Model')
plt.xticks(list(rf_orig_lc['Train sizes']), rotation = 90)
plt.legend(loc = "upper right")
plt.show()

##################
###RF EXTD PLOT###
##################

fig, ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(22)
ax.plot(rf_extd_lc['Train sizes'], rf_extd_lc['Test MSEs'], color = 'b', marker = 'o', linestyle = '--')
ax.set_xlabel('Train size')
ax.set_ylabel('MSE')
ax.set_title('Test MSEs per training size Random Forest, Extended Model')
plt.xticks(list(rf_extd_lc['Train sizes']), rotation = 90)
plt.legend(loc = "upper right")
plt.show()
