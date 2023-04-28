################################################
###FOURTH SCRIPT - REVERSING ONE-HOT-ENCODING###
################################################

import pandas as pd

'''
COMMENTS:

This is the fourth script in the sequence necessary to produce
the results in "What Makes a Satisfying Life? Prediction and 
Interpretation with Machine-Learning Algorithms", 
Gentile et al. (2022)    

As specified in the paper, when fitting and predicting with Random Forest
we considered the categorical variables as they are, without one-hot-encoding them,
id est without creating a dummy variable for each of the possible categories.

For this reason, we need to revert the one-hot-encoding procedure done in the
first script before creating the 100 train-test splits.
'''

path = 'C:\\some_local_path'

dest_path = 'C:\\some_local_path_noohed'

#Importing training sets

training_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + '\\train_' + j + '.csv'

    training_sets.append(pd.read_csv(import_path))

#Importing test set

test_sets = []

for i in range(1, 101):
    
    j = str(i)
    
    import_path = path + '\\test_' + j + '.csv'
        
    test_sets.append(pd.read_csv(import_path))
    
#Since we want to revert one-hot-encoding for all the posible categories of
#each categorical variable, we need to rename properly also those that we had
#dropped when running the linear regressions (those in the lists most_pop_cats 
#and least_pop_cats in the Second script - non-penalized linear regressions).

#We remind that these are either the most populous reference categories or
#the categories with less than 15 individuals, dropped to avoid collinearity
#issues with the non-penalized linear regressions.

#As usual, to simplify the for loops, recall that
#len(training_sets) == len(test_sets) == 100.

for i in range(0, len(training_sets)):
    
    training_sets[i].rename(columns = {'bd7ms_-7': 'Marital Status - Other missing',
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
                                       'bd7paliv_1': 'Whether father is alive - Yes'}, inplace = True)
    
    test_sets[i].rename(columns = {'bd7ms_-7': 'Marital Status - Other missing',
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
                                   'bd7paliv_1': 'Whether father is alive - Yes'
                                   }, inplace = True)

#HOW DOES THE REVERTING WORK?

#This is an explainer for the code below.

#First, we select only the dummies associated to each categorical 
#(i.e., all the dummies associated with Marital Status). This is done
#using a list comprehension [k for k ...] looking for all variables' 
#names with "Marital Status" in it, and then using it to subselect.

#For each individual, these columns will have value 0 but one of them, having 1.
#That is, for a married individuals all the dummies associated with being
#single, divorced, etc. will be 0, and the only 1 will be associated with
#the dummy for being married.

#Using then .idxmax(1) we therefore transform the previous dataframe
#to a series (one column only) containing for each individual what is her/his
#Marital Status.

#That is, if individual i is married, we move from having k - 1 columns
#of 0s and one only column with 1 ('Marital Status - Married'), to having
#one column only with the value 'Marital Status - Married' it it.

#Please refer to Pandas documentation on .idxmax() for additional examples.

#We finally proceed with simplyfying these labels excluding the whole 
#"Marital Status - " part, hence directly leaving only "Married", via
#a lambda function to substring each label, and then we recode the 
#variable using the original numeric values as per BCS documentation 
#using replace.

#All of the above are done for each categorical variable initially
#one-hot-encoded, and for each of the 100 train ant test sets.

for i in range(0, len(training_sets)):
    
    ############################
    ##REVERTING MARITAL STATUS##
    ############################
    
    ms = [k for k in list(training_sets[i]) if 'Marital Status' in k]
    
    orig_ms_train = training_sets[i][ms].idxmax(1)

    orig_ms_test = test_sets[i][ms].idxmax(1)
    
    orig_ms_train = pd.DataFrame(orig_ms_train.apply(lambda x: x[len('Marital Status - '):]))

    orig_ms_train.columns = ['Marital Status']
    
    #Transform 'Married in 1, etc..'
    
    orig_ms_train['Marital Status'].replace(['Married', 'Cohabiting', 'Single (never married)','Separated', 
                                             'Divorced','Widowed', 'Other missing'], 
                                            [1,2,3,4,5,6,-7], inplace = True)


    orig_ms_test = pd.DataFrame(orig_ms_test.apply(lambda x: x[len('Marital Status - '):]))

    orig_ms_test.columns = ['Marital Status']
    
    orig_ms_test['Marital Status'].replace(['Married', 'Cohabiting', 'Single (never married)','Separated', 
                                            'Divorced','Widowed', 'Other missing'], 
                                            [1,2,3,4,5,6,-7], inplace = True)
    
    ###################################
    ##REVERTING TYPE OF ACCOMMODATION##
    ###################################

    accom = [k for k in list(training_sets[i]) if 'Type of Accommodation' in k]

    orig_accom_train = training_sets[i][accom].idxmax(1)

    orig_accom_test = test_sets[i][accom].idxmax(1)

    orig_accom_train = pd.DataFrame(orig_accom_train.apply(lambda x: x[len('Type of Accommodation - '):]))

    orig_accom_train.columns = ['Type of Accommodation']
    
    orig_accom_train['Type of Accommodation'].replace(['A house or bungalow', 'Flat or Maisonette','Studio flat',
                                                       'A room / rooms','Something else','Not Applicable'],
                                                       [1,2,3,4,5,-1], inplace = True)
    
    orig_accom_test = pd.DataFrame(orig_accom_test.apply(lambda x: x[len('Type of Accommodation - '):]))

    orig_accom_test.columns = ['Type of Accommodation']
    
    orig_accom_test['Type of Accommodation'].replace(['A house or bungalow', 'Flat or Maisonette','Studio flat',
                                                      'A room / rooms','Something else','Not Applicable'],
                                                       [1,2,3,4,5,-1], inplace = True)
    
    ###########################
    ##REVERTING TENURE STATUS##
    ###########################

    ten2 = [k for k in list(training_sets[i]) if 'Tenure Status' in k]

    orig_ten2_train = training_sets[i][ten2].idxmax(1)

    orig_ten2_test = test_sets[i][ten2].idxmax(1)

    orig_ten2_train = pd.DataFrame(orig_ten2_train.apply(lambda x: x[len('Tenure Status - '):]))

    orig_ten2_train.columns = ['Tenure Status']
    
    orig_ten2_train['Tenure Status'].replace(['Own (outright)','Own - buying with help of a mortgage/loan', 
                                              'Pay part rent and part mortgage (shared/equity ownership)',                           
                                              'Rent it','Live here rent-free','Squatting', 'Other', 
                                              'Refusal', 'Do not Know'],
                                             [1,2,3,4,5,6,7,-9,-8], inplace = True)

    orig_ten2_test = pd.DataFrame(orig_ten2_test.apply(lambda x: x[len('Tenure Status - '):]))

    orig_ten2_test.columns = ['Tenure Status']
    
    orig_ten2_test['Tenure Status'].replace(['Own (outright)','Own - buying with help of a mortgage/loan', 
                                             'Pay part rent and part mortgage (shared/equity ownership)',                           
                                             'Rent it','Live here rent-free','Squatting', 'Other', 
                                             'Refusal', 'Do not Know'],
                                             [1,2,3,4,5,6,7,-9,-8], inplace = True)
    
    ###########################
    ##REVERTING MAIN ACTIVITY##
    ###########################

    ecact = [k for k in list(training_sets[i]) if 'Main Activity' in k]

    orig_ecact_train = training_sets[i][ecact].idxmax(1)

    orig_ecact_test = test_sets[i][ecact].idxmax(1)

    orig_ecact_train = pd.DataFrame(orig_ecact_train.apply(lambda x: x[len('Main Activity - '):]))

    orig_ecact_train.columns = ['Main Activity']
    
    orig_ecact_train['Main Activity'].replace(['Full-time paid employee', 'Looking after home/family',
                                               'Part-time paid employee (under 30 hours a week)',
                                               'Unemployed and seeking work', 'Full-time self-employed',
                                               'Part-time self-employed', 'Full-time education', 'Other',
                                               'Permanently sick/disabled', 'Temporarily sick/disabled',
                                               'On a government scheme for employment training', 'Do not know'],
                                                [1, 10, 2, 5, 3, 4, 6, 12, 9, 8, 7, -8], inplace = True)

    orig_ecact_test = pd.DataFrame(orig_ecact_test.apply(lambda x: x[len('Main Activity - '):]))

    orig_ecact_test.columns = ['Main Activity']
    
    orig_ecact_test['Main Activity'].replace(['Full-time paid employee', 'Looking after home/family',
                                              'Part-time paid employee (under 30 hours a week)',
                                              'Unemployed and seeking work', 'Full-time self-employed',
                                              'Part-time self-employed', 'Full-time education', 'Other',
                                              'Permanently sick/disabled', 'Temporarily sick/disabled',
                                              'On a government scheme for employment training', 'Do not know'],
                                              [1, 10, 2, 5, 3, 4, 6, 12, 9, 8, 7, -8], inplace = True)
    
    ############################################
    ##REVERTING HIGHEST ACADEMIC QUALIFICATION##
    ############################################

    hq13 = [k for k in list(training_sets[i]) if 'Highest Academic Qualification' in k]

    orig_hq13_train = training_sets[i][hq13].idxmax(1)

    orig_hq13_test = test_sets[i][hq13].idxmax(1)

    orig_hq13_train = pd.DataFrame(orig_hq13_train.apply(lambda x: x[len('Highest Academic Qualification - '):]))

    orig_hq13_train.columns = ['Highest Academic Qualification']
    
    orig_hq13_train['Highest Academic Qualification'].replace(['None', 'GCE A Level (or S Level)', 'GCSE', 'A/S Level',
                                                               'Degree (e.g. BA, BSc)', 'CSE', 'GCE O Level',
                                                               'Scottish School Certificate, Higher School Certificate',
                                                               'Higher degree (e.g. PhD, MSc)',
                                                               'Nursing or other para-medical qualification',
                                                               'PGCE-Post-graduate Certificate of Education',
                                                               'Diploma of Higher Education', 'Other teaching qualification',
                                                               'Other degree level qualification such as graduate membership',
                                                               'Do not know'],
                                                                [0, 6, 2, 4, 11, 1, 3, 5, 13, 7, 12, 9, 8, 10, -8], inplace = True)

    orig_hq13_test = pd.DataFrame(orig_hq13_test.apply(lambda x: x[len('Highest Academic Qualification - '):]))

    orig_hq13_test.columns = ['Highest Academic Qualification']
    
    orig_hq13_test['Highest Academic Qualification'].replace(['None', 'GCE A Level (or S Level)', 'GCSE', 'A/S Level',
                                                              'Degree (e.g. BA, BSc)', 'CSE', 'GCE O Level',
                                                              'Scottish School Certificate, Higher School Certificate',
                                                              'Higher degree (e.g. PhD, MSc)',
                                                              'Nursing or other para-medical qualification',
                                                              'PGCE-Post-graduate Certificate of Education',
                                                              'Diploma of Higher Education', 'Other teaching qualification',
                                                              'Other degree level qualification such as graduate membership',
                                                              'Do not know'],
                                                               [0, 6, 2, 4, 11, 1, 3, 5, 13, 7, 12, 9, 8, 10, -8], inplace = True)
    
    #########################################
    ##REVERTING WHETHER REGISTERED DISABLED##
    #########################################

    khldl2 = [k for k in list(training_sets[i]) if 'Whether Registered Disabled' in k]

    orig_khldl2_train = training_sets[i][khldl2].idxmax(1)

    orig_khldl2_test = test_sets[i][khldl2].idxmax(1)

    orig_khldl2_train = pd.DataFrame(orig_khldl2_train.apply(lambda x: x[len('Whether Registered Disabled - '):]))

    orig_khldl2_train.columns = ['Whether Registered Disabled']
    
    orig_khldl2_train['Whether Registered Disabled'].replace(['Yes', 'No but longterm disability',
                                                              'No and no longterm disability', 'Do not know'],
                                                              [1,2,3,-8], inplace = True)
    
    orig_khldl2_test = pd.DataFrame(orig_khldl2_test.apply(lambda x: x[len('Whether Registered Disabled - '):]))

    orig_khldl2_test.columns = ['Whether Registered Disabled']
    
    orig_khldl2_test['Whether Registered Disabled'].replace(['Yes', 'No but longterm disability',
                                                             'No and no longterm disability', 'Do not know'],
                                                             [1,2,3,-8], inplace = True)
    
    #######################################################
    ##REVERTING WHETHER HEALTH LIMITS EVERYDAY ACTIVITIES##
    #######################################################

    khllt = [k for k in list(training_sets[i]) if 'Whether health limits everyday activities' in k]

    orig_khllt_train = training_sets[i][khllt].idxmax(1)

    orig_khllt_test = test_sets[i][khllt].idxmax(1)

    orig_khllt_train = pd.DataFrame(orig_khllt_train.apply(lambda x: x[len('Whether health limits everyday activities - '):]))

    orig_khllt_train.columns = ['Whether health limits everyday activities']

    orig_khllt_train['Whether health limits everyday activities'].replace(['No and no health problems since last interview',
                                                                           'No but health problems since last interview', 
                                                                           'Yes'], [3,2,1], inplace = True)
                                                                           

    orig_khllt_test = pd.DataFrame(orig_khllt_test.apply(lambda x: x[len('Whether health limits everyday activities - '):]))

    orig_khllt_test.columns = ['Whether health limits everyday activities']
    
    orig_khllt_test['Whether health limits everyday activities'].replace(['No and no health problems since last interview',
                                                                          'No but health problems since last interview', 
                                                                          'Yes'], [3,2,1], inplace = True)
    
    ########################################
    ##REVERTING BMI WEIGHT STATUS CATEGORY##
    ########################################

    bmigp = [k for k in list(training_sets[i]) if 'BMI weight status category' in k]

    orig_bmigp_train = training_sets[i][bmigp].idxmax(1)

    orig_bmigp_test = test_sets[i][bmigp].idxmax(1)

    orig_bmigp_train = pd.DataFrame(orig_bmigp_train.apply(lambda x: x[len('BMI weight status category - '):]))

    orig_bmigp_train.columns = ['BMI weight status category']
    
    orig_bmigp_train.replace(['Normal (18.5-24.9)', 'Overweight (25-29.9)',
                              'Obese (30 and above)', 'Insufficient data',
                              'Underweight (< 18.5)'], [2,3,4,-7,1], inplace = True)
                              

    orig_bmigp_test = pd.DataFrame(orig_bmigp_test.apply(lambda x: x[len('BMI weight status category - '):]))

    orig_bmigp_test.columns = ['BMI weight status category']
    
    orig_bmigp_test.replace(['Normal (18.5-24.9)', 'Overweight (25-29.9)',
                             'Obese (30 and above)', 'Insufficient data',
                             'Underweight (< 18.5)'], [2,3,4,-7,1], inplace = True)
    
    ############################
    ##REVERTING SMOKING HABITS##
    ############################

    smoke = [k for k in list(training_sets[i]) if 'Smoking habits' in k]

    orig_smoke_train = training_sets[i][smoke].idxmax(1)

    orig_smoke_test = test_sets[i][smoke].idxmax(1)

    orig_smoke_train = pd.DataFrame(orig_smoke_train.apply(lambda x: x[len('Smoking habits - '):]))

    orig_smoke_train.columns = ['Smoking habits']
    
    orig_smoke_train['Smoking habits'].replace(['Up to 10 a day', '11 to 20 a day', 
                                                'Never smoked', 'Ex smoker',
                                                'Occasional smoker', 'More than 20 a day',
                                                'Daily but frequency not stated', 'Other missing'],
                                                [3,4,0,1,2,5,6,-7], inplace = True)

    orig_smoke_test = pd.DataFrame(orig_smoke_test.apply(lambda x: x[len('Smoking habits - '):]))

    orig_smoke_test.columns = ['Smoking habits']
    
    orig_smoke_test['Smoking habits'].replace(['Up to 10 a day', '11 to 20 a day', 
                                               'Never smoked', 'Ex smoker',
                                               'Occasional smoker', 'More than 20 a day',
                                               'Daily but frequency not stated', 'Other missing'],
                                               [3,4,0,1,2,5,6,-7], inplace = True)
    
    #################################################
    ##REVERTING ALCOHOL UNITS IN A WEEK BY CATEGORY##
    #################################################

    alcohol = [k for k in list(training_sets[i]) if 'Alcohol units in a week by category' in k]

    orig_alcohol_train = training_sets[i][alcohol].idxmax(1)

    orig_alcohol_test = test_sets[i][alcohol].idxmax(1)

    orig_alcohol_train = pd.DataFrame(orig_alcohol_train.apply(lambda x: x[len('Alcohol units in a week by category - '):]))

    orig_alcohol_train.columns = ['Alcohol units in a week by category']
    
    orig_alcohol_train['Alcohol units in a week by category'].replace(['1 to 14', '15 to 21', 
                                                                       'Never drinks or only on special occasions','None reported',
                                                                       '22 to 39', 'More than 39'],
                                                                        [1,2,-1,0,3,4], inplace = True)

    orig_alcohol_test = pd.DataFrame(orig_alcohol_test.apply(lambda x: x[len('Alcohol units in a week by category - '):]))

    orig_alcohol_test.columns = ['Alcohol units in a week by category']
    
    orig_alcohol_test['Alcohol units in a week by category'].replace(['1 to 14', '15 to 21', 
                                                                      'Never drinks or only on special occasions','None reported',
                                                                      '22 to 39', 'More than 39'],
                                                                       [1,2,-1,0,3,4], inplace = True)
    
    #####################################
    ##REVERTING WHETHER MOTHER IS ALIVE##
    #####################################

    malive = [k for k in list(training_sets[i]) if 'Whether mother is alive' in k]

    orig_malive_train = training_sets[i][malive].idxmax(1)

    orig_malive_test = test_sets[i][malive].idxmax(1)

    orig_malive_train = pd.DataFrame(orig_malive_train.apply(lambda x: x[len('Whether mother is alive - '):]))

    orig_malive_train.columns = ['Whether mother is alive']

    orig_malive_train['Whether mother is alive'].replace(['Yes in household', 'Yes', 
                                                          'No', 'No reported dead last sweep',
                                                          'Missing', 'Do not know'], [0,1,2,3,-8,-7], inplace = True)

    orig_malive_test = pd.DataFrame(orig_malive_test.apply(lambda x: x[len('Whether mother is alive - '):]))

    orig_malive_test.columns = ['Whether mother is alive']
    
    orig_malive_test['Whether mother is alive'].replace(['Yes in household', 'Yes', 
                                                         'No', 'No reported dead last sweep',
                                                         'Missing', 'Do not know'], [0,1,2,3,-8,-7], inplace = True)
    
    #####################################
    ##REVERTING WHETHER FATHER IS ALIVE##
    #####################################

    palive = [k for k in list(training_sets[i]) if 'Whether father is alive' in k]

    orig_palive_train = training_sets[i][palive].idxmax(1)

    orig_palive_test = test_sets[i][palive].idxmax(1)

    orig_palive_train = pd.DataFrame(orig_palive_train.apply(lambda x: x[len('Whether father is alive - '):]))

    orig_palive_train.columns = ['Whether father is alive']
    
    orig_palive_train['Whether father is alive'].replace(['Yes in household', 'Yes', 
                                                          'No', 'No reported dead last sweep',
                                                          'Do not know', 'Missing'], [0,1,2,3,-8,-7], inplace = True)

    orig_palive_test = pd.DataFrame(orig_palive_test.apply(lambda x: x[len('Whether father is alive - '):]))

    orig_palive_test.columns = ['Whether father is alive']
    
    orig_palive_test['Whether father is alive'].replace(['Yes in household', 'Yes', 
                                                         'No', 'No reported dead last sweep',
                                                         'Do not know', 'Missing'], [0,1,2,3,-8,-7], inplace = True)
    
    ################################
    ##FINAL CLEANING AND APPENDING##
    ################################
    
    #We finally delete the now useless dummies and concatenate back the
    #correct categorical variables, stored in orig_ms_train, orig_ms_test, etc.
    
    dummies_to_drop = ms + accom + ten2 + ecact + hq13 + khldl2 + khllt + bmigp + smoke + alcohol + malive + palive

    training_sets[i].drop(dummies_to_drop, axis = 1, inplace = True) 
                                                
    test_sets[i].drop(dummies_to_drop, axis = 1, inplace = True) 
    
    training_sets[i] = pd.concat([training_sets[i],
                                 orig_ms_train,
                                 orig_accom_train,
                                 orig_ten2_train,
                                 orig_ecact_train,
                                 orig_hq13_train,
                                 orig_khldl2_train,
                                 orig_khllt_train,
                                 orig_bmigp_train,
                                 orig_smoke_train,
                                 orig_alcohol_train,
                                 orig_malive_train,
                                 orig_palive_train],
                                 axis = 1)
                                            
    test_sets[i] = pd.concat([test_sets[i],
                             orig_ms_test,
                             orig_accom_test,
                             orig_ten2_test,
                             orig_ecact_test,
                             orig_hq13_test,
                             orig_khldl2_test,
                             orig_khllt_test,
                             orig_bmigp_test,
                             orig_smoke_test,
                             orig_alcohol_test,
                             orig_malive_test,
                             orig_palive_test],
                             axis = 1)
    
    j = str(i + 1)

    dest_path_i_train = dest_path + '\\train_noohed_' + j + '.csv'    
    
    dest_path_i_test = dest_path + '\\test_noohed_' + j + '.csv'   
    
    training_sets[i].to_csv(dest_path_i_train)
    
    test_sets[i].to_csv(dest_path_i_test)

