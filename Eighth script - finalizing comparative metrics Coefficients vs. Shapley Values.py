##################################################################
###EIGHTH SCRIPT: FINAIZING COMPARATIVE METRICS, COEFFS vs. SVs###
##################################################################

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

'''
COMMENTS:
    
This is the eighth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

In particular, this is the third and last of those necessary to produce
the comparative table of Linear Regressions' Coefficients and
Random Forests' Shapley Values.

Here, we first import the Shapley Values for both the Original and Extended
models, as well as the Non-Penalized Linear Regression Coefficients for
the Original and the Ridge Regression Coefficients for the Extended, respectively
produced in the sixt and seventh scripts.

Producing Table 9 - the comparison for the Original model - is straightforward,
since no additional computations are needed. Conversely, some additional
steps are needed to produce Table 10 - the comparison for the Extended model -
in particular regarding Equations 15 and 16.

Everything is done only on Train - Test Split 1.
''' 

import_path_dset = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits\\'

import_and_dest_path_coeffs = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\'

train_1 = pd.read_csv(import_path_dset + 'train_1.csv')

coeffs_extd = pd.read_csv(import_and_dest_path_coeffs + 'Train_1_joined_ridge_coeffs_all.csv')

coeffs_orig = pd.read_csv(import_and_dest_path_coeffs + 'Train_1_orig_linreg_coeffs.csv')

SVs_extd = pd.read_csv(import_and_dest_path_coeffs + 'Average_MASVs_joined.csv')

SVs_orig = pd.read_csv(import_and_dest_path_coeffs + 'Average_MASVs_orig.csv')

for i in [SVs_extd, SVs_orig]:
    
    i.drop(['Unnamed: 0'], axis = 1, inplace = True)
    
for j in [coeffs_extd, coeffs_orig]:
    
    j.rename(columns = {'Unnamed: 0': 'Feature'}, inplace = True)

######################
####ORIGINAL MODEL####
######################

#In the Original model, we simply need to first compute the MASVs
#from the Shapley Values, and then inner join the table with the 
#Coefficients' dataframe.
    
SVs_orig.loc['MASV'] = SVs_orig.abs().mean()

SVs_orig_T = SVs_orig.T

SVs_orig_T_1 = SVs_orig_T[['MASV']]

SVs_orig_T_1.reset_index(inplace = True)

SVs_orig_T_1.rename(columns = {'index': 'Feature'}, inplace = True)

Metric_origs = SVs_orig_T_1.merge(coeffs_orig, how = 'inner', on = 'Feature')

Metric_origs.to_csv(import_and_dest_path_coeffs + 'SV_coeffs_orig.csv')

######################
####EXTENDED MODEL####
######################

#In the Extended model, more operations are needed: they are thoroughly
#described.

#We start with some due renaming of variables.

coeffs_extd.rename(columns = {"1": "coef"}, inplace = True)

varnames_to_replace = {'bd7ms_-7': 'Marital Status - Other missing',
                       'bd7ms_1': 'Marital Status - Married',
                       'bd7ms_6': 'Marital Status - Widowed',
                       'b7accom_1': 'Type of Accommodation - A house or bungalow',
                       'b7accom_4': 'Type of Accommodation - A room / rooms',
                       'b7ten2_-8': 'Tenure Status - Do not Know',
                       'b7ten2_2': 'Tenure Status - Own - buying with help of a mortgage/loan',
                       'b7ten2_6': 'Tenure Status - Squatting',
                       'bd7ecact_-8': 'Main Activity - Do not know',
                       'bd7ecact_1': 'Main Activity - Full-time paid employee',
                       'bd7ecact_7': 'Main Activity - On a government scheme for employment training',
                       'bd7hq13_-8': 'Highest Academic Qualification - Do not know',
                       'bd7hq13_3': 'Highest Academic Qualification - GCE O Level',
                       'b7khldl2_2': 'Whether Registered Disabled - No but longterm disability',
                       'b7khldl2_-8': 'Whether Registered Disabled - Do not know',
                       'b7khllt_2': 'Whether health limits everyday activities - No but health problems since last interview',
                       'bd7bmigp_2': 'BMI weight status category - Normal (18.5-24.9)',
                       'bd7smoke_-7': 'Smoking habits - Other missing',
                       'bd7smoke_0': 'Smoking habits - Never smoked',
                       'bd7smoke_6': 'Smoking habits - Daily but frequency not stated',
                       'bd7dgrp_1': 'Alcohol units in a week by category - 1 to 14',
                       'bd7maliv_-8': 'Whether mother is alive - Do not know',
                       'bd7maliv_1': 'Whether mother is alive - Yes',
                       'bd7paliv_1': 'Whether father is alive - Yes'}

train_1.drop(['lifesatisfaction'], axis = 1, inplace = True)

train_1.rename(columns = varnames_to_replace, inplace = True)

#We delete the coefficient for the constant, as we do not have an equivalent in Shapley Values.

coeffs_extd = coeffs_extd[coeffs_extd['Feature'] != '(Intercept)']

#We now create the list containing the list of the coefficients actually
#estimated. We do not have ridge coefficients for 'Educational Achievement', 'Employed', 
#'Has a Partner', and 'Any long-standing illness'.

#Since we have 

for_naming_coeffs_extd = ['Educational Achievement', 'Employed', 'Has a Partner', 'Any long-standing illness']

to_clean_coeffs_extd = [x for x in list(train_1) if x not in for_naming_coeffs_extd]

#In looking at coeffs_extd's column "Feature", and of the list 
#to_clean_coeffs_extd, it can be seen that they have the same order in the
#names of the variables. 

#Therefore, we simply rename the values in coeffs_extd['Feature'] with the 
#appropriate complete names in to_clean_coeffs_extd.

coeffs_extd['Feature'] = coeffs_extd['Feature'].replace(list(coeffs_extd['Feature'].unique()), to_clean_coeffs_extd)

#Now, the Feature names in coeffs_extd and in train_1 are identical.

#Now, I can select, in train_1, those columns in which I need to do the coefficients-building
#operation, equations 15 and 16 in the paper.

to_remove = ['Log Income','Good Conduct','Female',
             'Number of people in the household',
             'Number of natural children in the household',
             'Number of non-natural children in the household',
             'Number of rooms in the household',
             'Physical Health',
             'Emotional Health',
             'Any long-standing illness',
             'Educational Achievement',
             'Employed',
             'Has a Partner']

to_average = [x for x in list(train_1) if x not in to_remove]

train_1_to_average = train_1[to_average]

del to_average, to_remove, for_naming_coeffs_extd, to_clean_coeffs_extd

#The remaining variables in train_1_to_average are those on which we need
#to calculate chi_{j,l} as in equation 15. The following for loop does it.

proportions = []

for i in list(train_1_to_average):
            
    proportion = np.sum(train_1_to_average[i] == 1)/len(train_1_to_average)
        
    print(i, proportion)
        
    proportions.append([i, proportion])
    
proportions_df = pd.DataFrame(proportions, columns = ['Feature', 'Proportion'] )

#We now inner join this dataframe including the proportions of 1s for each
#category for each categorical variable (the aformentioned chi_{j,l} as in 
#equation 15) with the dataframe including the associated coefficients.

#In this manner, we can compute the product chi_{j,l} * |beta_{j,l}| for each
#category j of each categorical variable l.

coeffs_proportions = coeffs_extd.merge(proportions_df, how = 'inner', on = 'Feature')

coeffs_proportions.rename(columns = {'s1': 'Coefficient'}, inplace = True)

coeffs_proportions['|Coefficient|'] = coeffs_proportions['Coefficient'].apply(lambda x: abs(x))

coeffs_proportions['|Coefficient| * Proportion'] = coeffs_proportions['|Coefficient|'] * coeffs_proportions['Proportion']

#and now, to finalize the Derived Coefficients as in equation 16, we sum
#the products chi_{j,l} * |beta_{j,l}|, available in the column 
#'|Coefficient| * Proportion' of coeffs_proportions,
#across all the the k different categories for each categorical variable.

#We do the operation separatedly for each different categorical variable.
#Considering Marital Status as showing example:

ms_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Marital Status')]

ms_df.loc['Sum(|Coefficient| * Proportion)'] = ms_df['|Coefficient| * Proportion'].sum()

#In other words, we first subset coeffs_proportions so to keep only those 
#'|Coefficient| * Proportion' associated to variables containing the string
#"Marital Status", id est 'Marital Status - Married', 'Marital Status - Single',
#etc. Then we sum them, obtaining the value of equation 16 for Marital Status.

accom_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Type of Accommodation')]

accom_df.loc['Sum(|Coefficient| * Proportion)'] = accom_df['|Coefficient| * Proportion'].sum()

ten_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Tenure Status')]

ten_df.loc['Sum(|Coefficient| * Proportion)'] = ten_df['|Coefficient| * Proportion'].sum()

main_act_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Main Activity')]

main_act_df.loc['Sum(|Coefficient| * Proportion)'] = main_act_df['|Coefficient| * Proportion'].sum()

haq_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Highest Academic Qualification')]

haq_df.loc['Sum(|Coefficient| * Proportion)'] = haq_df['|Coefficient| * Proportion'].sum()

whether_dis_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Whether Registered Disabled')]

whether_dis_df.loc['Sum(|Coefficient| * Proportion)'] = whether_dis_df['|Coefficient| * Proportion'].sum()

whether_lim_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Whether health limits everyday activities')]

whether_lim_df.loc['Sum(|Coefficient| * Proportion)'] = whether_lim_df['|Coefficient| * Proportion'].sum()

bmi_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('BMI weight status category')]

bmi_df.loc['Sum(|Coefficient| * Proportion)'] = bmi_df['|Coefficient| * Proportion'].sum()

smoke_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Smoking habits')]

smoke_df.loc['Sum(|Coefficient| * Proportion)'] = smoke_df['|Coefficient| * Proportion'].sum()

alcohol_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Alcohol units in a week by category')]

alcohol_df.loc['Sum(|Coefficient| * Proportion)'] = alcohol_df['|Coefficient| * Proportion'].sum()

mother_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Whether mother is alive')]

mother_df.loc['Sum(|Coefficient| * Proportion)'] = mother_df['|Coefficient| * Proportion'].sum()

father_df = coeffs_proportions[coeffs_proportions['Feature'].str.contains('Whether father is alive')]

father_df.loc['Sum(|Coefficient| * Proportion)'] = father_df['|Coefficient| * Proportion'].sum()

#All the following operations are simple data wrangling to create the final
#Table 10, including both the just-derived coefficients (the underlined ones
#in the aformentioned Table 10) and the ones for the continuous numerical 
#variables directly extracted from the Ridge Regression.

SVs_extd.loc['MASV'] = SVs_extd.abs().mean()

SVs_extd_T = SVs_extd.T

SVs_extd_T_1 = SVs_extd_T[['MASV']]

SVs_extd_T_1.reset_index(inplace = True)

SVs_extd_T_1.rename(columns = {'index': 'Feature'}, inplace = True)

#Extracting the derived coefficients

ms_coeff = pd.DataFrame(['Marital Status', ms_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

accom_coeff = pd.DataFrame(['Type of Accommodation', accom_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

ten_coeff = pd.DataFrame(['Tenure Status', ten_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

main_act_coeff = pd.DataFrame(['Main Activity', main_act_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

haq_coeff = pd.DataFrame(['Highest Academic Qualification', haq_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

whether_dis_coeff = pd.DataFrame(['Whether Registered Disabled', whether_dis_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

whether_lim_coeff = pd.DataFrame(['Whether health limits everyday activities', whether_lim_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

bmi_coeff = pd.DataFrame(['BMI weight status category', bmi_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

smoke_coeff = pd.DataFrame(['Smoking habits', smoke_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

alcohol_coeff = pd.DataFrame(['Alcohol units in a week by category', alcohol_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

mother_coeff = pd.DataFrame(['Whether mother is alive', mother_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

father_coeff = pd.DataFrame(['Whether father is alive', father_df.loc['Sum(|Coefficient| * Proportion)'][0]], index = ['Feature', 'coef']).T

#and the original ridge coefficients

ln_inc_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Log Income'])
ln_inc_coeff.rename(columns = {"s1": "coef"}, inplace = True)

crime_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Good Conduct'])
crime_coeff.rename(columns = {"s1": "coef"}, inplace = True)

female_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Female'])
female_coeff.rename(columns = {"s1": "coef"}, inplace = True)

numhh_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Number of people in the household'])
numhh_coeff.rename(columns = {"s1": "coef"}, inplace = True)

nchhh_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Number of natural children in the household'])
nchhh_coeff.rename(columns = {"s1": "coef"}, inplace = True)

ochhh_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Number of non-natural children in the household'])
ochhh_coeff.rename(columns = {"s1": "coef"}, inplace = True)

numrms_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Number of rooms in the household'])
numrms_coeff.rename(columns = {"s1": "coef"}, inplace = True)

phy_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Physical Health'])
phy_coeff.rename(columns = {"s1": "coef"}, inplace = True)

mhindex_coeff = pd.DataFrame(coeffs_extd[coeffs_extd['Feature'] == 'Emotional Health'])
mhindex_coeff.rename(columns = {"s1": "coef"}, inplace = True)

coeffs_to_concat = [ms_coeff, accom_coeff, ten_coeff, main_act_coeff, haq_coeff,
                    whether_dis_coeff, whether_lim_coeff, bmi_coeff, smoke_coeff,
                    alcohol_coeff, mother_coeff, father_coeff, ln_inc_coeff,
                    crime_coeff, female_coeff, numhh_coeff, nchhh_coeff, 
                    ochhh_coeff, numrms_coeff, phy_coeff,
                    mhindex_coeff]

extd_feat_coeffs = pd.concat(coeffs_to_concat, ignore_index = True)

extd_feat_coeffs.columns = ['Feature', 'Coefficients']

Metric_extd = SVs_extd_T_1.merge(extd_feat_coeffs, how = 'inner', on = 'Feature')

Metric_extd.to_csv(import_and_dest_path_coeffs + 'SV_coeffs_extd.csv')
