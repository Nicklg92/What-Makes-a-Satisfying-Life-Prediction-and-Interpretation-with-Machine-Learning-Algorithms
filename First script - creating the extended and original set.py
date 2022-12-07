############################################################
###FIRST SCRIPT - CREATING THE ORIGINAL AND EXTENDED SETS###
############################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import randint

scaler = StandardScaler()

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

'''
COMMENTS:

This is the first script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022).

Here, in particular, we proceed with adding the variables
reported in the British Cohort Study (BCS, henceforth) documentation 
as "Some Key Variables", and we add them - including necessary cleaning - 
to the original ones, used to produce the results in “What Predicts a
Successful Life? A Life-Course Model of Well-Being”, 
Layard et al. (2014).

These codes will be particularly rich in comments, to ensure 
readability also to less experienced Python coders.

If you have more experience, you can skip the parts explaining
the codes.
'''

import_path = "C:\\Users\\niccolo.gentile\\Desktop\\BCS\\"

followup = pd.read_csv(import_path + "bcs_2004_followup.tab", sep = "\t")

#followup.shape 

#9665 x 2638

Some_key_variables = ['bd7sex', 'bd7ethnic', 'bd7ms',
                      'bd7spphh', 'bd7numhh', 'bd7nchhh', 'bd7ochhh',
                      'b7accom', 'b7numrms', 'b7ten2', 'bd7ecact',
                      'b7nssec', 'bd7ns8', 'b7seg', 'b7sc', 'bd7potha',
                      'bd7hq13', 'bd7hq5', 'b7khlstt', 'b7khldl2',
                      'b7lsiany', 'b7khllt', 'bd7bmigp', 'bd7smoke',
                      'bd7dgrp', 'bd7maliv', 'bd7paliv', 'b7othrea', 
                      'b7lifet1', 'bcsid']

followup_skv = followup[Some_key_variables]


#################################################
####EXPLORATORY DATA ANALYSIS: MISSING VALUES####
#################################################

#How many unique values for each variable?

followup_skv.nunique()

#How many missing values for each variable?

#followup_skv.lt(0).sum()

#followup_skv.b7lifet1.lt(0).sum() 

#71

#As by practice, we start deleting the 71 individuals for whom
#life satisfaction (dependent variables) is missing.

#Missingness, in BCS, is denoted with negative values.

followup_skv = followup_skv.loc[followup_skv['b7lifet1'] >= 0, :]

#For the subsequent computations, since bcsid is not numeric, we temporarily
#split the datasets:

bcsid = followup_skv['bcsid']

followup_skv = followup_skv.drop(['bcsid'], axis = 1)

#In each variable, there are possibly multiple different negative values.
#This is to reflect the reason for the missingness, as explained
#below.

#What are the most frequent missing value type for each of these 
#variables?

neg_for_var = []

for i in list(followup_skv):
    
    #We iterate here across all the columns in followup_skv.
    #In each of them, we first check what are all the possible
    #unique values, making this array a list - followup_skv[i].unique().tolist().
    #Then, in this list, we see what are the possible negative values,
    #using a list comprehension: x for x in colvar_uniques if x < 0
    #In the end, neg_for_var will be a list of lists, each list
    #reporting which are the possible different negative values
    #(hence kind of missings) for each variables in followup_skv.
    #The final step - dict(zip(list(followup_skv), neg_for_var)) -
    #creates a dictionary - 'key': value set of pairs - associating the
    #list of possible negative values to each variable.
    
    colvar_uniques = followup_skv[i].unique().tolist()

    #neg_for_var.append(list(filter(lambda x: (x < 0), colvar_uniques)))
    
    #The above is an alternative to the list comprehension
    #used below.
        
    neg_for_var.append([x for x in colvar_uniques if x < 0])

which_neg_vals = dict(zip(list(followup_skv), neg_for_var))

#For what the type of different missing associated with each
#negative value, please refer to the original BCS documentation,
#namely bcs_2004_followup_ukda_data_dictionary.

#The overall point is that, both in these variables as well as
#on the more general dataset, all the missings are not at random.

#Some removals of variables are performed. 

#For references to all these specific variables are, please refer
#to the original BCS documentation, 
#namely bcs_2004_followup_ukda_data_dictionary.
  
#We keep only bd7ns8, dropping b7nssec,
#b7seg and b7sc.
    
followup_skv_2 = followup_skv.drop(['b7nssec', 'b7seg', 'b7sc'],
                                   axis = 1)   

#Also, bd7hq13 already comprehends bd7hq5.

followup_skv_3 = followup_skv_2.drop(['bd7hq5'], axis = 1) 

for i in list(followup_skv_3):
    
    print(i, followup_skv_3[i].unique())
  
###############################################
##EXPLORATORY DATA ANALYSIS: ONE HOT ENCODING##
###############################################

#Categorical variables need to be one-hot-encoded (id est,
#each category has to be transformed in a dummy variable).

#How many variables need to be one - hot - encoded?

followup_skv.nunique()

for i in list(followup_skv):
    
    print(i, followup_skv[i].unique())

#The variables that are getting encoded are:

#bd7ethnic, ethnicity
#bd7ms, marital status
#b7accom, type of accommodation
#b7ten2, tenure status
#bd7ecact, main activity
#bd7ns8, NS-SEC 8 analytic version
#bd7potha, partner's / spouse's main activity
#bd7hq13, highest qualification
#b7khlstt, self-assessed health
#b7khldl2, whether registered disabled
#b7khllt, whether health limits everyday activities
#bd7bmigp, Body Mass Index (measured in bands)
#bd7smoke, smoking habits (measured in bands)
#bd7dgrp, alcohol units in a week by category
#bd7maliv, whether mother is alive (including categories if in household)
#bd7maliv, whether father is alive (including categories if in household)

#We remind the reader to bcs_2004_followup_ukda_data_dictionary for a more
#detailed description of all the above.
    
vars_to_ohe = ['bd7ethnic', 'bd7ms', 'b7accom', 'b7ten2', 'bd7ecact',
               'bd7ns8', 'bd7potha', 'bd7hq13', 'b7khlstt', 'b7khldl2',
               'b7khllt', 'bd7bmigp', 'bd7smoke', 'bd7dgrp', 'bd7maliv',
               'bd7paliv']


ohc_datasets = []

#The function pd.get_dummies() creates dummy variables for
#each of the values in the in the pandas Series passed
#as first argument.
    
#pd.get_dummies() has also an additional parameter, "dummy_na", which
#can be either True or False dependending on whether we want to create
#a missing-indicator column. 
    
#In our case, however, the missing values are NOT indicated by
#nans - as can be easily observed since followup_skv.isna().sum().sum()
#returns 0 - but rather by NEGATIVE values, which are already
#considered by the function to create the dummies.
    
#In other words, if you have missing values as nans in your 
#categorical variabes, and you want to create a missing indicator,
#you need to put dummy_na = True. If the missing values are 
#labelled in some other manner (as our case) or you don't
#want to create a missing indicator, you can put dummy_na = False).
    
#In our case, it would just create a set of all 0s columns.

for i in vars_to_ohe:

    ohc_datasets.append(pd.get_dummies(followup_skv_3[i], prefix = i, dummy_na = False))
    
ohe_followup_skv_4 = pd.concat(ohc_datasets, axis = 1)

ohe_followup_skv_5 = pd.concat([ohe_followup_skv_4, 
                                followup_skv_3.drop(vars_to_ohe, axis = 1)],
                                axis = 1)

for i in list(ohe_followup_skv_5):
    
    print(i, ohe_followup_skv_5[i].unique())
    
#There is still the problem of the missings in the non to one - hot - encoded
#variables. How to deal with them?
    
ohe_followup_skv_5.lt(0).sum()

#We drop b7othrea, whether is currently in a non-residential relationship.

ohe_followup_skv_6 = ohe_followup_skv_5.drop(['b7othrea'], axis = 1) 

#The treatment of the missing values is specified in the paper.
#The imputation of the missing values in the continuous
#variables is performed before fitting and predicting via Linear 
#Regressions, whereas they are left as they are when doing
#the same with Random Forests. The degree of missingness
#in these is however negligible - between 0.3% and 0.6% of 
#the observations - and different treatments of them don't change the results.

for i in list(ohe_followup_skv_6):
    
    print(i, ohe_followup_skv_6[i].unique())
    
#We finally harmonize the remaining dummies.
    
#Note that b7lsiany = 2 <=> No long - standing illness, disability or infirmity, 
#bd7sex = 2 <=> Female.

#The variables are recoded so that: 
#bd7sex = 1 <=> Female  
#b7lsiany = 0 <=> No long - standing illness, disability or infirmity.
        
ohe_followup_skv_6['bd7sex'] = ohe_followup_skv_6['bd7sex'].apply(lambda x: x - 1)

ohe_followup_skv_6['b7lsiany'] = ohe_followup_skv_6['b7lsiany'].apply(lambda x: 0 if x == 2 else 1)

for i in list(ohe_followup_skv_6):
    
    print(i, ohe_followup_skv_6[i].unique())
    
ohe_followup_skv_6.shape #9594 x 148

#To simplify the subsequent computations, we separate dependent
#and independent variables.

ohe_followup_skv_indvars = ohe_followup_skv_6.drop(['b7lifet1'], axis = 1)

ls = ohe_followup_skv_6['b7lifet1']

#######################################
###JOINING WITH THE RESTRICTED MODEL###
#######################################

#The joining operations with the Original data are divided in three steps:
    
#A) Left joining ohe_followup_skv_indvars with the physical health at 26
#from Clark and Lepinteur (2019).

#B) Left join the resulting dataset from A) with mental health at 26
#from Layard et al. (2014).

#C) Inner joining the resulting dataset from B) with all the remaining
#Original variables from Layard et al. (2014).

#The first two joining operations were "left" instead of "inner"
#to avoid losing too many individuals. This means that 
#we accept reasonable degrees of missingness in physical and
#mental health in order to retain information about the
#other variables. 

#Anyway, the inner join at the third step guarantees a final dataset
#with 665 missings only in mental health, as reported in Section 2 of the paper.  

############################################################
##A) PHYSICAL HEALTH AT 26 FROM CLARK AND LEPINTEUR (2019)##
############################################################

#As specified in section 2 of the paper, in the Original data,
#we considered a measure of health different from the original
#subjective of Layard et al. (2014).

#More precisely, we considered the number of conditions from which the 
#individual suffers as in Clark and Lepinteur (2019), 
#“The Causes and Consequences of Early-adult Unemployment:
#Evidence from Cohort Data”, Journal of Economic Behavior & 
#Organization, 166, 107–124. This variable represents the count 
#of conditions presented in Appendix D. It is measured
#at age 26.

phy26 = pd.read_stata(import_path + 'phy26.dta')

phy26.rename(columns = {'bcsid_70': 'bcsid'}, inplace = True)

#phy26.shape
#(17196, 2). The two columns are indeed the ID and the physical health.
#There are however more individuals than in our data.

#Notice that:
#phy26.isna().sum().sum()
#0

#and:
    
#phy26.phy_h.lt(0).sum()
#0

#as well, meaning that it is never missing.

#We remind that the pandas Series bcsid was created in line 86.
#Since no row deletion have taken place since then, the 
#ids are still aligned with the independent variables as they were in that line.

ohe_followup_skv_indvars_ids = pd.concat([ohe_followup_skv_indvars, bcsid],
                                         axis = 1) 

#We therefore have ids both in ohe_followup_skv_indvars_ids and phy26,
#which allows us to join the two datasets.

skv_indvars_ph26 = pd.merge(ohe_followup_skv_indvars_ids, phy26, on = 'bcsid', how = 'left') 

#len(skv_indvars_ph26) == len(ohe_followup_skv_indvars_ids) = 9594

#The resulting dataset from a let join has as many rows as the left one.

#Did the joining operation led to some nans?

#skv_indvars_ph26.isna().sum().sum()

#694. More precisely:
    
#skv_indvars_ph26.isna().sum()

#694 in physical health only. How come?

#Notice that:

#len(set(list(ohe_followup_skv_indvars_ids['bcsid'])) - set(list(phy26['bcsid'])))
#is 694, meaning that there are 694 IDs in ohe_followup_skv_indvars_ids that are not 
#available in phy26. 

#That is, 694 of the individuals in the dataset we have created so far are 
#not included in phy26 from Clark and Lepinteur (2019). 

#Clearly, for all of them physical health will be missing at 26.

###########################################################
##B) MHINDEX AT 26 FROM THE ORIGINAL LAYARD ET AL. (2014)##
###########################################################

#We now introduce the Original dataset with all the features of Layard et al. (2014).
#Considering them, on top of the aformenetioned physical health in Clark and Lepinteur (2019),
#we can create the Original dataset.

#As reported in the paper, Mental Health is considered at age 26. For the 665 individuals
#for whom is missing, we considered the value at 30.

#In bcs_clean_SF.dta, mhindex is reported as a lagged measure in the variable mhindex_1.
#Hence, the values of mhindex_1 for someone who's 30 represent her/his mental health
#when she/he was 26. We import here the values at 26.
#We also import the IDs so to be able to join with skv_indvars_ph26,
#the dataset we have created thus far.

all_vars = pd.read_stata(import_path + 'bcs_clean_SF.dta')

all_vars_30 = all_vars.loc[(all_vars['age'] == 30)] 
   
all_vars_30_id_mh = all_vars_30[['bcsid_70', 'mhindex_1']]

all_vars_30_id_mh.columns = ['bcsid', 'mhindex_26']

#Notice that:
#all_vars_30_id_mh.mhindex_26.lt(0).sum()
#0

#and:

#all_vars_30_id_mh.isnull().sum()
#0

#meaning that in the Original data by Layard et al. (2014), mental health was never missing at 26. 

#However:

#len(all_vars_30_id_mh)

#10251

#meaning that the two datasets do not necessarily have the same individuals, and some
#missingness may be generated by joining it with skv_indvars_ph26 created thus far.

skv_indvars_ph26_mh26 = pd.merge(skv_indvars_ph26, all_vars_30_id_mh, on = 'bcsid', how = 'left')

#Did the joining operation led to some nans?

#skv_indvars_ph26_mh26.isna().sum().sum()

#2086. More precisely:
    
#skv_indvars_ph26_mh26.isna().sum()

#694 in phy_h and 1392 mhindex_26. How come?

#We can do the same check we did when joining phy26 and ohe_followup_skv_indvars_ids.

#Namely:

#len(set(list(skv_indvars_ph26['bcsid'])) - set(list(all_vars_30_id_mh['bcsid'])))
#is 1392, meaning that there are 1392 IDs in skv_indvars_ph26 that are not 
#available in all_vars_30_id_mh. Id est, 1392 of the individuals in the dataset we have created
#thus far are  not included in the original BCS from Layard et al. (2014) at 30. 

#Clearly, for all of them mental health will be missing at 26.

#We clean up the global environment, for better readability
#in the Variable Explorer in Spyder:

del all_vars_30, all_vars_30_id_mh, colvar_uniques, followup
del followup_skv, followup_skv_2, followup_skv_3, bcsid, neg_for_var
del ohc_datasets, ohe_followup_skv_4, ohe_followup_skv_5, ohe_followup_skv_6
del ohe_followup_skv_indvars, ohe_followup_skv_indvars_ids, phy26
del skv_indvars_ph26, Some_key_variables, vars_to_ohe, which_neg_vals

####################################################################
##C) FINAL JOIN WITH THE OTHER ORIGINALS FROM LAYARD ET AL. (2014)##
####################################################################

#In the final step, we join the dataset we have produced so far
#with the other Original variables from Layard et al. (2014)
#at 34.

#We manually recreate the composite Ed_Achiev and Has a Partner
#as specified in their Appendix A. Please refer to it to check
#where the specific values come from.

orig_vars = all_vars[all_vars['age'] == 34]

orig_vars = orig_vars.assign(Ed_Achiev = np.where(orig_vars['phd30'] == 1, 0.75,
                                                  np.where(orig_vars['degree30'] == 1, 0.486,
                                                           np.where(orig_vars['alevel30'] == 1, 0.237,
                                                                    np.where(orig_vars['gcse30'] == 1, 0.188,
                                                                             np.where(orig_vars['cse30'] == 1, 0.043,
                                                                                      0))))))

orig_vars = orig_vars.assign(Has_a_partner = np.where((orig_vars['married'] == 1) & (orig_vars['has_child'] == 1), 0.685,
                                                      np.where((orig_vars['married'] == 1) & (orig_vars['has_child'] == 0), 0.530,
                                                               np.where(orig_vars['single_withchildren'] == 1, -0.004, 0))))       

#We also import mhindex_1, that in this case refers to mental
#health at 30 since we have filtered for age 34.

#We will use the values in this variable to impute the missings
#in mhindex_26 as specified in the paper.                                                                                             

orig_vars = orig_vars[['lifesatisfaction', 'ln_income', 
                       'Ed_Achiev', 'unemp',
                       'Has_a_partner', 'crime', 'mhindex_1',
                       'female', 'id_num']]

orig_vars.rename(columns = {'id_num': 'bcsid'}, inplace = True)

#Before joining, notice that:

#ls.index == skv_indvars_ph26_mh26.index
#False

#Always good practice to check for this before using pandas' concat:
#when the indexes are mismatched, concat creates NaNs (see
#pandas' documentation to see why).

ls.reset_index(drop = True, inplace = True)

skv_indvars_ph26_mh26.reset_index(drop = True, inplace = True)

skv_indvars_ph26_mh26_ls = pd.concat([ls, skv_indvars_ph26_mh26], axis = 1)

#In this case, we finally go for an inner join. 

orig_and_skv = orig_vars.merge(skv_indvars_ph26_mh26_ls, on = 'bcsid', how = 'inner')

#orig_and_skv.isna().sum()

#only 665 missing values in mhindex_26, which we impute
#using the value at 30.

orig_and_skv['mhindex_26'] = np.where(orig_and_skv['mhindex_26'].isnull() == True, 
                                      orig_and_skv['mhindex_1'], 
                                      orig_and_skv['mhindex_26'])

#We split X and y for the subsequent train - test splits.

y = orig_and_skv['lifesatisfaction']

#some indvars are repeated, since available both in 
#the dataset from Layard et al.(2014), as well as in our newly added ones.
#This is the case for gender ('bd7sex'), and life satisfaction itself.

#We also drop the no-longer needed mental health at 30.

X = orig_and_skv.drop(['lifesatisfaction', 'bcsid', 'b7lifet1', 'bd7sex', 'mhindex_1'],
                                         axis = 1)

#Some additional variables are dropped, main reason being the
#degree of sparsity they induce, in terms of 0s.
#Too many dummies.

always_to_drop = ['bd7ethnic_-7', 'bd7ecact_11', 'bd7potha_14', 'bd7ethnic_1',
                  'bd7ethnic_2','bd7ethnic_3','bd7ethnic_4','bd7ethnic_5', 
                  'bd7ethnic_6','bd7ethnic_7','bd7ethnic_8',
                  'bd7ethnic_9', 'bd7ethnic_10','bd7ethnic_11', 'bd7ethnic_12',
                  'bd7ethnic_13','bd7ethnic_14','bd7ethnic_15','bd7ethnic_16',
                  'bd7ns8_1','bd7ns8_2','bd7ns8_3','bd7ns8_4',
                  'bd7ns8_5','bd7ns8_6','bd7ns8_7', 'bd7spphh', 
                  'b7khlstt_1', 'b7khlstt_2','b7khlstt_3','b7khlstt_4', 'b7khlstt_5',
                  'bd7potha_1',  'bd7potha_2', 'bd7potha_3', 'bd7potha_4',
                  'bd7potha_5', 'bd7potha_6', 'bd7potha_7', 'bd7potha_8',
                  'bd7potha_9', 'bd7potha_10', 'bd7potha_11', 'bd7potha_12', 
                  'bd7potha_14','bd7potha_15',  'bd7potha_16','bd7potha_17', 
                  'bd7potha_94', 'bd7potha_95', 'bd7ns8_-3', 'bd7ns8_-1', 'b7khlstt_-8',
                  'bd7potha_-8', 'bd7potha_-1']

X.drop(always_to_drop, axis = 1, inplace = True)

#For one person, gender value is .48175734.

X['female'] = round(X['female'], 0)

X.rename(columns = {'ln_income': 'Log Income',
                    'crime': 'Good Conduct',
                    'female': 'Female',
                    'bd7ms_2': 'Marital Status - Cohabiting',
                    'bd7ms_3': 'Marital Status - Single (never married)',
                    'bd7ms_4': 'Marital Status - Separated',
                    'bd7ms_5': 'Marital Status - Divorced',
                    'b7accom_-1': 'Type of Accommodation - Not Applicable',
                    'b7accom_2': 'Type of Accommodation - Flat or Maisonette',
                    'b7accom_3': 'Type of Accommodation - Studio flat',
                    'b7accom_5': 'Type of Accommodation - Something else',
                    'b7ten2_-9': 'Tenure Status - Refusal',
                    'b7ten2_1': 'Tenure Status - Own (outright)',
                    'b7ten2_3': 'Tenure Status - Pay part rent and part mortgage (shared/equity ownership)',
                    'b7ten2_4': 'Tenure Status - Rent it',
                    'b7ten2_5': 'Tenure Status - Live here rent-free',
                    'b7ten2_7': 'Tenure Status - Other',
                    'bd7ecact_2': 'Main Activity - Part-time paid employee (under 30 hours a week)',
                    'bd7ecact_3': 'Main Activity - Full-time self-employed',
                    'bd7ecact_4': 'Main Activity - Part-time self-employed',
                    'bd7ecact_5': 'Main Activity - Unemployed and seeking work',
                    'bd7ecact_6': 'Main Activity - Full-time education',
                    'bd7ecact_8': 'Main Activity - Temporarily sick/disabled',
                    'bd7ecact_9': 'Main Activity - Permanently sick/disabled',
                    'bd7ecact_10': 'Main Activity - Looking after home/family',
                    'bd7ecact_12': 'Main Activity - Other',
                    'bd7hq13_0': 'Highest Academic Qualification - None',
                    'bd7hq13_1': 'Highest Academic Qualification - CSE',
                    'bd7hq13_2': 'Highest Academic Qualification - GCSE',
                    'bd7hq13_4': 'Highest Academic Qualification - A/S Level',
                    'bd7hq13_5': 'Highest Academic Qualification - Scottish School Certificate, Higher School Certificate',
                    'bd7hq13_6': 'Highest Academic Qualification - GCE A Level (or S Level)',
                    'bd7hq13_7': 'Highest Academic Qualification - Nursing or other para-medical qualification',
                    'bd7hq13_8': 'Highest Academic Qualification - Other teaching qualification',
                    'bd7hq13_9': 'Highest Academic Qualification - Diploma of Higher Education',
                    'bd7hq13_10': 'Highest Academic Qualification - Other degree level qualification such as graduate membership',
                    'bd7hq13_11': 'Highest Academic Qualification - Degree (e.g. BA, BSc)',
                    'bd7hq13_12': 'Highest Academic Qualification - PGCE-Post-graduate Certificate of Education',
                    'bd7hq13_13': 'Highest Academic Qualification - Higher degree (e.g. PhD, MSc)',
                    'b7khldl2_1': 'Whether Registered Disabled - Yes',
                    'b7khldl2_3': 'Whether Registered Disabled - No and no longterm disability',
                    'b7khllt_1': 'Whether health limits everyday activities - Yes',
                    'b7khllt_3': 'Whether health limits everyday activities - No and no health problems since last interview',
                    'bd7bmigp_-7': 'BMI weight status category - Insufficient data',
                    'bd7bmigp_1': 'BMI weight status category - Underweight (< 18.5)',
                    'bd7bmigp_3': 'BMI weight status category - Overweight (25-29.9)',
                    'bd7bmigp_4': 'BMI weight status category - Obese (30 and above)',
                    'bd7smoke_1': 'Smoking habits - Ex smoker',
                    'bd7smoke_2': 'Smoking habits - Occasional smoker',
                    'bd7smoke_3': 'Smoking habits - Up to 10 a day',
                    'bd7smoke_4': 'Smoking habits - 11 to 20 a day',
                    'bd7smoke_5': 'Smoking habits - More than 20 a day',
                    'bd7dgrp_-1': 'Alcohol units in a week by category - Never drinks or only on special occasions',
                    'bd7dgrp_0': 'Alcohol units in a week by category - None reported',
                    'bd7dgrp_2': 'Alcohol units in a week by category - 15 to 21',
                    'bd7dgrp_3': 'Alcohol units in a week by category - 22 to 39',
                    'bd7dgrp_4': 'Alcohol units in a week by category - More than 39',
                    'bd7maliv_-7': 'Whether mother is alive - Missing',
                    'bd7maliv_0': 'Whether mother is alive - Yes in household',
                    'bd7maliv_2': 'Whether mother is alive - No',
                    'bd7maliv_3': 'Whether mother is alive - No reported dead last sweep',
                    'bd7paliv_-8': 'Whether father is alive - Do not know',
                    'bd7paliv_-7': 'Whether father is alive - Missing',
                    'bd7paliv_0': 'Whether father is alive - Yes in household',
                    'bd7paliv_2': 'Whether father is alive - No',
                    'bd7paliv_3': 'Whether father is alive - No reported dead last sweep',
                    'bd7numhh': 'Number of people in the household',
                    'bd7nchhh': 'Number of natural children in the household',
                    'bd7ochhh': 'Number of non-natural children in the household',
                    'b7numrms': 'Number of rooms in the household',
                    'b7lsiany': 'Any long-standing illness',
                    'phy_h': 'Physical Health',
                    'mhindex_26': 'Emotional Health', 
                    'unemp': 'Employed',
                    'Has_a_partner': 'Has a Partner',
                    'Ed_Achiev': 'Educational Achievement'}, inplace = True)

#Continuous variables are standardized.
    
to_stand = ['Log Income', 'Good Conduct', 'Number of people in the household',
            'Number of natural children in the household', 'Number of non-natural children in the household',
            'Number of rooms in the household', 'Physical Health', 'Emotional Health',
            'Has a Partner', 'Educational Achievement']

#As how reported in the paper, for 'Number of people in the household',
#'Number of natural children in the household', 'Number of non-natural children in the household',
#and 'Number of rooms in the household', we impute the missing values (negatives) with
#mean. Given their scarcity, no changes were observed throughout
#the algorhtms without doing it.

numvars = ['Number of people in the household','Number of natural children in the household', 
           'Number of non-natural children in the household','Number of rooms in the household']
    
X[X[numvars] < 0] = np.nan

for j in numvars:

    X[j].fillna(X[j].mean(), inplace = True)
    
    X[j] = round(X[j])
    
dest_path = 'C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits'  

np.random.seed(42)

for i in range(1,101):
    
    seed = randint(0, 1000)    
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        test_size = 0.20,
                                                        random_state = seed)    
    
    X_train[to_stand] = scaler.fit_transform(X_train[to_stand])
        
    X_test[to_stand] = scaler.transform(X_test[to_stand])
    
    train = pd.concat([y_train, X_train], axis = 1)
    
    test = pd.concat([y_test, X_test], axis = 1)
        
    save_train = dest_path + '\\train_' + str(i) + '.csv'
    
    save_test = dest_path + '\\test_' + str(i) + '.csv'
    
    train.to_csv(save_train, index = False)
    
    test.to_csv(save_test, index = False)
