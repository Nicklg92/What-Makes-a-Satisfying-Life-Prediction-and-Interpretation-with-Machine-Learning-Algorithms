################################
###THIRD SCRIPT: ELASTIC NETS###
################################

library(glmnet)
library(coefplot)

options(scipen=999)

#COMMENTS:

#This it the third script  in the sequence necessary to produce
#the results in "What Makes a Satisfying Life? Prediction and 
#Interpretation with Machine-Learning Algorithms", 
#Gentile et al. (2022).

#Here, in particular, we proceed with running the penalized
#regressions - Ridge, LASSO and Elastic Nets - on the 100 train-test splits generated in the
#first script.

rm(list = ls())

path <- "C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits\\"

#########################
###IMPORTING THE FILES###
#########################

#Importing training sets

training_sets <- list()

for (i in 1:100){
  
  j <- as.character(i)
  
  import_path <- paste0(path, "train_", j, ".csv")
  
  training_sets[[i]] <- read.csv(paste0(import_path))
  
  names(training_sets[[i]]) <- gsub(x = names(training_sets[[i]]), pattern = "\\.", replacement = " ")
  
}

#Importing test sets

test_sets <- list()

for (i in 1:100){
  
  j <- as.character(i)
  
  import_path <- paste0(path, "test_", j, ".csv")
  
  test_sets[[i]] <- read.csv(paste0(import_path))
  
  names(test_sets[[i]]) <- gsub(x = names(test_sets[[i]]), pattern = "\\.", replacement = " ")
  
}

for (i in 1:100){
  
  training_sets[[i]]["Physical Health"] <- -1 * training_sets[[i]]["Physical Health"]
  
  test_sets[[i]]["Physical Health"] <- -1 * test_sets[[i]]["Physical Health"]
  
}  

######################
###ORIGINAL DATASET###
######################

MSEs_test_ridge_orig <- list()

MSEs_train_ridge_orig <- list()

MSEs_test_lasso_orig <- list()

MSEs_train_lasso_orig <- list()

MSEs_test_en025_orig <- list()

MSEs_train_en025_orig <- list()

MSEs_test_en050_orig <- list()

MSEs_train_en050_orig <- list()

MSEs_test_en075_orig <- list()

MSEs_train_en075_orig <- list()

fitted_ridge_orig <- list()

fitted_lasso_orig <- list()

fitted_en025_orig <- list()

fitted_en050_orig <- list()

fitted_en075_orig <- list()

set.seed(42)

for (i in 1:100){
  
  X_train_orig <- training_sets[[i]][,which(names(training_sets[[i]]) %in% c('Log Income', 'Educational Achievement', 'Employed', 
                                                                             'Good Conduct', 'Female', 'Has a Partner', 'Physical Health',
                                                                             'Emotional Health'))]
  
  y_train_orig <- training_sets[[i]]["lifesatisfaction"]
  
  X_test_orig <- test_sets[[i]][,which(names(test_sets[[i]]) %in% c('Log Income', 'Educational Achievement', 'Employed', 
                                                                    'Good Conduct', 'Female', 'Has a Partner', 'Physical Health',
                                                                    'Emotional Health'))]
  
  y_test_orig <- test_sets[[i]]["lifesatisfaction"]
  
  ##################
  ##RIDGE ORIGINAL##
  ##################
  
  #We leave standardize = TRUE so that also the dummies
  #get standardized before running the penalized 
  #regressions (necessary as specified in the paper).
  
  cvfit_ridge_orig <- cv.glmnet(x = as.matrix(X_train_orig),
                                y = as.matrix(y_train_orig),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 0)
  
  fitted_test_ridge_orig <- predict(cvfit_ridge_orig, newx = as.matrix(X_test_orig), s = "lambda.min")
  
  MSE_test_ridge_orig <- colMeans((y_test_orig - fitted_test_ridge_orig)^(2))
  
  fitted_training_ridge_orig <- predict(cvfit_ridge_orig, newx = as.matrix(X_train_orig), s = "lambda.min")
  
  MSE_train_ridge_orig <- colMeans((y_train_orig - fitted_training_ridge_orig)^(2))
  
  fitted_ridge_orig[[i]] <- cvfit_ridge_orig
  
  MSEs_train_ridge_orig[[i]] <- MSE_train_ridge_orig
  
  MSEs_test_ridge_orig[[i]] <- MSE_test_ridge_orig
  
  ##################
  ##LASSO ORIGINAL##
  ##################
  
  cvfit_lasso_orig <- cv.glmnet(x = as.matrix(X_train_orig),
                                y = as.matrix(y_train_orig),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 1)
  
  fitted_test_lasso_orig <- predict(cvfit_lasso_orig, newx = as.matrix(X_test_orig), s = "lambda.min")
  
  MSE_test_lasso_orig <- colMeans((y_test_orig - fitted_test_lasso_orig)^(2))
  
  fitted_training_lasso_orig <- predict(cvfit_lasso_orig, newx = as.matrix(X_train_orig), s = "lambda.min")
  
  MSE_train_lasso_orig <- colMeans((y_train_orig - fitted_training_lasso_orig)^(2))
  
  fitted_lasso_orig[[i]] <- cvfit_lasso_orig
  
  MSEs_train_lasso_orig[[i]] <- MSE_train_lasso_orig
  
  MSEs_test_lasso_orig[[i]] <- MSE_test_lasso_orig
  
  ##################
  ##EN025 ORIGINAL##
  ##################
  
  cvfit_en025_orig <- cv.glmnet(x = as.matrix(X_train_orig),
                                y = as.matrix(y_train_orig),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 0.25)
  
  fitted_test_en025_orig <- predict(cvfit_en025_orig, newx = as.matrix(X_test_orig), s = "lambda.min")
  
  MSE_test_en025_orig <- colMeans((y_test_orig - fitted_test_en025_orig)^(2))
  
  fitted_training_en025_orig <- predict(cvfit_en025_orig, newx = as.matrix(X_train_orig), s = "lambda.min")
  
  MSE_train_en025_orig <- colMeans((y_train_orig - fitted_training_en025_orig)^(2))
  
  fitted_en025_orig[[i]] <- cvfit_en025_orig
  
  MSEs_train_en025_orig[[i]] <- MSE_train_en025_orig
  
  MSEs_test_en025_orig[[i]] <- MSE_test_en025_orig
  
  ##################
  ##EN050 ORIGINAL##
  ##################
  
  cvfit_en050_orig <- cv.glmnet(x = as.matrix(X_train_orig),
                                y = as.matrix(y_train_orig),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 0.50)
  
  fitted_test_en050_orig <- predict(cvfit_en050_orig, newx = as.matrix(X_test_orig), s = "lambda.min")
  
  MSE_test_en050_orig <- colMeans((y_test_orig - fitted_test_en050_orig)^(2))
  
  fitted_training_en050_orig <- predict(cvfit_en050_orig, newx = as.matrix(X_train_orig), s = "lambda.min")
  
  MSE_train_en050_orig <- colMeans((y_train_orig - fitted_training_en050_orig)^(2))
  
  fitted_en050_orig[[i]] <- cvfit_en050_orig
  
  MSEs_train_en050_orig[[i]] <- MSE_train_en050_orig
  
  MSEs_test_en050_orig[[i]] <- MSE_test_en050_orig
  
  ##################
  ##EN075 ORIGINAL##
  ##################
  
  cvfit_en075_orig <- cv.glmnet(x = as.matrix(X_train_orig),
                                y = as.matrix(y_train_orig),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 0.75)
  
  fitted_test_en075_orig <- predict(cvfit_en075_orig, newx = as.matrix(X_test_orig), s = "lambda.min")
  
  MSE_test_en075_orig <- colMeans((y_test_orig - fitted_test_en075_orig)^(2))
  
  fitted_training_en075_orig <- predict(cvfit_en075_orig, newx = as.matrix(X_train_orig), s = "lambda.min")
  
  MSE_train_en075_orig <- colMeans((y_train_orig - fitted_training_en075_orig)^(2))
  
  fitted_en075_orig[[i]] <- cvfit_en075_orig
  
  MSEs_train_en075_orig[[i]] <- MSE_train_en075_orig
  
  MSEs_test_en075_orig[[i]] <- MSE_test_en075_orig
  
}

#################################
###100 ORIGINAL RIDGEs RESULTS###
#################################

MSEs_train_ridge_orig_vec <- unlist(MSEs_train_ridge_orig)

MSEs_train_ridge_orig_vec_mean <- mean(MSEs_train_ridge_orig_vec)

MSEs_train_ridge_orig_vec_std <- sd(MSEs_train_ridge_orig_vec)

MSEs_test_ridge_orig_vec <- unlist(MSEs_test_ridge_orig)

MSEs_test_ridge_orig_vec_mean <- mean(MSEs_test_ridge_orig_vec)

MSEs_test_ridge_orig_vec_std <- sd(MSEs_test_ridge_orig_vec)

fitted_lambdas_ridge_orig <- list()

for (i in 1:100){
  
  fitted_lambdas_ridge_orig[[i]] <- fitted_ridge_orig[[i]]$lambda.min
  
} 

fitted_lambdas_ridge_orig_vec <- unlist(fitted_lambdas_ridge_orig)

fitted_lambdas_ridge_orig_vec_mean <- mean(fitted_lambdas_ridge_orig_vec)

fitted_lambdas_ridge_orig_vec_std <- sd(fitted_lambdas_ridge_orig_vec)

#################################
###100 ORIGINAL LASSOs RESULTS###
#################################

MSEs_train_lasso_orig_vec <- unlist(MSEs_train_lasso_orig)

MSEs_train_lasso_orig_vec_mean <- mean(MSEs_train_lasso_orig_vec)

MSEs_train_lasso_orig_vec_std <- sd(MSEs_train_lasso_orig_vec)

MSEs_test_lasso_orig_vec <- unlist(MSEs_test_lasso_orig)

MSEs_test_lasso_orig_vec_mean <- mean(MSEs_test_lasso_orig_vec)

MSEs_test_lasso_orig_vec_std <- sd(MSEs_test_lasso_orig_vec)

fitted_lambdas_lasso_orig <- list()

for (i in 1:100){
  
  fitted_lambdas_lasso_orig[[i]] <- fitted_lasso_orig[[i]]$lambda.min
  
} 

fitted_lambdas_lasso_orig_vec <- unlist(fitted_lambdas_lasso_orig)

fitted_lambdas_lasso_orig_vec_mean <- mean(fitted_lambdas_lasso_orig_vec)

fitted_lambdas_lasso_orig_vec_std <- sd(fitted_lambdas_lasso_orig_vec)

fitted_nonzeros_lasso_orig <- list()

for (i in 1:100){
  
  fitted_nonzeros_lasso_orig[[i]] <- nrow(extract.coef(fitted_lasso_orig[[i]]))
  
} 

fitted_nonzeros_lasso_orig_vec <- unlist(fitted_nonzeros_lasso_orig)

fitted_nonzeros_lasso_orig_vec_mean <- mean(fitted_nonzeros_lasso_orig_vec)

fitted_nonzeros_lasso_orig_vec_std <- sd(fitted_nonzeros_lasso_orig_vec)

##################################
####100 ORIGINAL EN25s RESULTS####
##################################

MSEs_train_en025_orig_vec <- unlist(MSEs_train_en025_orig)

MSEs_train_en025_orig_vec_mean <- mean(MSEs_train_en025_orig_vec)

MSEs_train_en025_orig_vec_std <- sd(MSEs_train_en025_orig_vec)

MSEs_test_en025_orig_vec <- unlist(MSEs_test_en025_orig)

MSEs_test_en025_orig_vec_mean <- mean(MSEs_test_en025_orig_vec)

MSEs_test_en025_orig_vec_std <- sd(MSEs_test_en025_orig_vec)

fitted_lambdas_en025_orig <- list()

for (i in 1:100){
  
  fitted_lambdas_en025_orig[[i]] <- fitted_en025_orig[[i]]$lambda.min
  
} 

fitted_lambdas_en025_orig_vec <- unlist(fitted_lambdas_en025_orig)

fitted_lambdas_en025_orig_vec_mean <- mean(fitted_lambdas_en025_orig_vec)

fitted_lambdas_en025_orig_vec_std <- sd(fitted_lambdas_en025_orig_vec)

fitted_nonzeros_en025_orig <- list()

for (i in 1:100){
  
  fitted_nonzeros_en025_orig[[i]] <- nrow(extract.coef(fitted_en025_orig[[i]]))
  
} 

fitted_nonzeros_en025_orig_vec <- unlist(fitted_nonzeros_en025_orig)

fitted_nonzeros_en025_orig_vec_mean <- mean(fitted_nonzeros_en025_orig_vec)

fitted_nonzeros_en025_orig_vec_std <- sd(fitted_nonzeros_en025_orig_vec)

##################################
####100 ORIGINAL EN50s RESULTS####
##################################

MSEs_train_en050_orig_vec <- unlist(MSEs_train_en050_orig)

MSEs_train_en050_orig_vec_mean <- mean(MSEs_train_en050_orig_vec)

MSEs_train_en050_orig_vec_std <- sd(MSEs_train_en050_orig_vec)

MSEs_test_en050_orig_vec <- unlist(MSEs_test_en050_orig)

MSEs_test_en050_orig_vec_mean <- mean(MSEs_test_en050_orig_vec)

MSEs_test_en050_orig_vec_std <- sd(MSEs_test_en050_orig_vec)

fitted_lambdas_en050_orig <- list()

for (i in 1:100){
  
  fitted_lambdas_en050_orig[[i]] <- fitted_en050_orig[[i]]$lambda.min
  
} 

fitted_lambdas_en050_orig_vec <- unlist(fitted_lambdas_en050_orig)

fitted_lambdas_en050_orig_vec_mean <- mean(fitted_lambdas_en050_orig_vec)

fitted_lambdas_en050_orig_vec_std <- sd(fitted_lambdas_en050_orig_vec)

fitted_nonzeros_en050_orig <- list()

for (i in 1:100){
  
  fitted_nonzeros_en050_orig[[i]] <- nrow(extract.coef(fitted_en050_orig[[i]]))
  
} 

fitted_nonzeros_en050_orig_vec <- unlist(fitted_nonzeros_en050_orig)

fitted_nonzeros_en050_orig_vec_mean <- mean(fitted_nonzeros_en050_orig_vec)

fitted_nonzeros_en050_orig_vec_std <- sd(fitted_nonzeros_en050_orig_vec)

##################################
####100 ORIGINAL EN75s RESULTS####
##################################

MSEs_train_en075_orig_vec <- unlist(MSEs_train_en075_orig)

MSEs_train_en075_orig_vec_mean <- mean(MSEs_train_en075_orig_vec)

MSEs_train_en075_orig_vec_std <- sd(MSEs_train_en075_orig_vec)

MSEs_test_en075_orig_vec <- unlist(MSEs_test_en075_orig)

MSEs_test_en075_orig_vec_mean <- mean(MSEs_test_en075_orig_vec)

MSEs_test_en075_orig_vec_std <- sd(MSEs_test_en075_orig_vec)

fitted_lambdas_en075_orig <- list()

for (i in 1:100){
  
  fitted_lambdas_en075_orig[[i]] <- fitted_en075_orig[[i]]$lambda.min
  
} 

fitted_lambdas_en075_orig_vec <- unlist(fitted_lambdas_en075_orig)

fitted_lambdas_en075_orig_vec_mean <- mean(fitted_lambdas_en075_orig_vec)

fitted_lambdas_en075_orig_vec_std <- sd(fitted_lambdas_en075_orig_vec)

fitted_nonzeros_en075_orig <- list()

for (i in 1:100){
  
  fitted_nonzeros_en075_orig[[i]] <- nrow(extract.coef(fitted_en075_orig[[i]]))
  
} 

fitted_nonzeros_en075_orig_vec <- unlist(fitted_nonzeros_en075_orig)

fitted_nonzeros_en075_orig_vec_mean <- mean(fitted_nonzeros_en075_orig_vec)

fitted_nonzeros_en075_orig_vec_std <- sd(fitted_nonzeros_en075_orig_vec)

#########################
#########################
#########################

####################
###JOINED DATASET###
####################

MSEs_test_ridge_joined <- list()

MSEs_train_ridge_joined <- list()

MSEs_test_lasso_joined <- list()

MSEs_train_lasso_joined <- list()

MSEs_test_en025_joined <- list()

MSEs_train_en025_joined <- list()

MSEs_test_en050_joined <- list()

MSEs_train_en050_joined <- list()

MSEs_test_en075_joined <- list()

MSEs_train_en075_joined <- list()

fitted_ridge_joined <- list()

fitted_lasso_joined <- list()

fitted_en025_joined <- list()

fitted_en050_joined <- list()

fitted_en075_joined <- list()

for (i in 1:100){
  
  #Differently from the non-penalized regressions, in
  #this case, as specified in the paper, we keep
  #in all the dummies, since penalized regressions
  #automatically deal with multicollinearity.
  
  #We therefore drop only 'Educational Achievement', 'Employed', 
  #'Has a Partner', and "Any long standing illness", for the
  #reasons specified in the paper and in the comments in
  #the non-penalized linear regression's script.
  
  X_train_joined <- training_sets[[i]][,-which(names(training_sets[[i]]) %in% c('Educational Achievement', 'Employed',
                                                                                'Has a Partner', 'lifesatisfaction', "Any long standing illness"))]
  
  y_train_joined <- training_sets[[i]]["lifesatisfaction"]
  
  X_test_joined <- test_sets[[i]][,-which(names(test_sets[[i]]) %in% c('Educational Achievement', 'Employed', 
                                                                       'Has a Partner', 'lifesatisfaction', "Any long standing illness"))]
  
  y_test_joined <- test_sets[[i]]["lifesatisfaction"]
  
  ################
  ##RIDGE JOINED##
  ################
  
  cvfit_ridge_joined <- cv.glmnet(x = as.matrix(X_train_joined),
                                  y = as.matrix(y_train_joined),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 0)
  
  fitted_test_ridge_joined <- predict(cvfit_ridge_joined, newx = as.matrix(X_test_joined), s = "lambda.min")
  
  MSE_test_ridge_joined <- colMeans((y_test_joined - fitted_test_ridge_joined)^(2))
  
  fitted_training_ridge_joined <- predict(cvfit_ridge_joined, newx = as.matrix(X_train_joined), s = "lambda.min")
  
  MSE_train_ridge_joined <- colMeans((y_train_joined - fitted_training_ridge_joined)^(2))
  
  fitted_ridge_joined[[i]] <- cvfit_ridge_joined
  
  MSEs_train_ridge_joined[[i]] <- MSE_train_ridge_joined
  
  MSEs_test_ridge_joined[[i]] <- MSE_test_ridge_joined
  
  ################
  ##LASSO JOINED##
  ################
  
  cvfit_lasso_joined <- cv.glmnet(x = as.matrix(X_train_joined),
                                  y = as.matrix(y_train_joined),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 1)
  
  fitted_test_lasso_joined <- predict(cvfit_lasso_joined, newx = as.matrix(X_test_joined), s = "lambda.min")
  
  MSE_test_lasso_joined <- colMeans((y_test_joined - fitted_test_lasso_joined)^(2))
  
  fitted_training_lasso_joined <- predict(cvfit_lasso_joined, newx = as.matrix(X_train_joined), s = "lambda.min")
  
  MSE_train_lasso_joined <- colMeans((y_train_joined - fitted_training_lasso_joined)^(2))
  
  fitted_lasso_joined[[i]] <- cvfit_lasso_joined
  
  MSEs_train_lasso_joined[[i]] <- MSE_train_lasso_joined
  
  MSEs_test_lasso_joined[[i]] <- MSE_test_lasso_joined
  
  ################
  ##EN025 JOINED##
  ################
  
  cvfit_en025_joined <- cv.glmnet(x = as.matrix(X_train_joined),
                                  y = as.matrix(y_train_joined),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 0.25)
  
  fitted_test_en025_joined <- predict(cvfit_en025_joined, newx = as.matrix(X_test_joined), s = "lambda.min")
  
  MSE_test_en025_joined <- colMeans((y_test_joined - fitted_test_en025_joined)^(2))
  
  fitted_training_en025_joined <- predict(cvfit_en025_joined, newx = as.matrix(X_train_joined), s = "lambda.min")
  
  MSE_train_en025_joined <- colMeans((y_train_joined - fitted_training_en025_joined)^(2))
  
  fitted_en025_joined[[i]] <- cvfit_en025_joined
  
  MSEs_train_en025_joined[[i]] <- MSE_train_en025_joined
  
  MSEs_test_en025_joined[[i]] <- MSE_test_en025_joined
  
  ################
  ##EN050 JOINED##
  ################
  
  cvfit_en050_joined <- cv.glmnet(x = as.matrix(X_train_joined),
                                  y = as.matrix(y_train_joined),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 0.50)
  
  fitted_test_en050_joined <- predict(cvfit_en050_joined, newx = as.matrix(X_test_joined), s = "lambda.min")
  
  MSE_test_en050_joined <- colMeans((y_test_joined - fitted_test_en050_joined)^(2))
  
  fitted_training_en050_joined <- predict(cvfit_en050_joined, newx = as.matrix(X_train_joined), s = "lambda.min")
  
  MSE_train_en050_joined <- colMeans((y_train_joined - fitted_training_en050_joined)^(2))
  
  fitted_en050_joined[[i]] <- cvfit_en050_joined
  
  MSEs_train_en050_joined[[i]] <- MSE_train_en050_joined
  
  MSEs_test_en050_joined[[i]] <- MSE_test_en050_joined
  
  ################
  ##EN075 JOINED##
  ################
  
  cvfit_en075_joined <- cv.glmnet(x = as.matrix(X_train_joined),
                                  y = as.matrix(y_train_joined),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 0.75)
  
  fitted_test_en075_joined <- predict(cvfit_en075_joined, newx = as.matrix(X_test_joined), s = "lambda.min")
  
  MSE_test_en075_joined <- colMeans((y_test_joined - fitted_test_en075_joined)^(2))
  
  fitted_training_en075_joined <- predict(cvfit_en075_joined, newx = as.matrix(X_train_joined), s = "lambda.min")
  
  MSE_train_en075_joined <- colMeans((y_train_joined - fitted_training_en075_joined)^(2))
  
  fitted_en075_joined[[i]] <- cvfit_en075_joined
  
  MSEs_train_en075_joined[[i]] <- MSE_train_en075_joined
  
  MSEs_test_en075_joined[[i]] <- MSE_test_en075_joined
  
}

###############################
###100 JOINED RIDGEs RESULTS###
###############################

MSEs_train_ridge_joined_vec <- unlist(MSEs_train_ridge_joined)

MSEs_train_ridge_joined_vec_mean <- mean(MSEs_train_ridge_joined_vec)

MSEs_train_ridge_joined_vec_std <- sd(MSEs_train_ridge_joined_vec)

MSEs_test_ridge_joined_vec <- unlist(MSEs_test_ridge_joined)

MSEs_test_ridge_joined_vec_mean <- mean(MSEs_test_ridge_joined_vec)

MSEs_test_ridge_joined_vec_std <- sd(MSEs_test_ridge_joined_vec)

fitted_lambdas_ridge_joined <- list()

for (i in 1:100){
  
  fitted_lambdas_ridge_joined[[i]] <- fitted_ridge_joined[[i]]$lambda.min
  
} 

fitted_lambdas_ridge_joined_vec <- unlist(fitted_lambdas_ridge_joined)

fitted_lambdas_ridge_joined_vec_mean <- mean(fitted_lambdas_ridge_joined_vec)

fitted_lambdas_ridge_joined_vec_std <- sd(fitted_lambdas_ridge_joined_vec)

###############################
###100 JOINED LASSOs RESULTS###
###############################

MSEs_train_lasso_joined_vec <- unlist(MSEs_train_lasso_joined)

MSEs_train_lasso_joined_vec_mean <- mean(MSEs_train_lasso_joined_vec)

MSEs_train_lasso_joined_vec_std <- sd(MSEs_train_lasso_joined_vec)

MSEs_test_lasso_joined_vec <- unlist(MSEs_test_lasso_joined)

MSEs_test_lasso_joined_vec_mean <- mean(MSEs_test_lasso_joined_vec)

MSEs_test_lasso_joined_vec_std <- sd(MSEs_test_lasso_joined_vec)

fitted_lambdas_lasso_joined <- list()

for (i in 1:100){
  
  fitted_lambdas_lasso_joined[[i]] <- fitted_lasso_joined[[i]]$lambda.min
  
} 

fitted_lambdas_lasso_joined_vec <- unlist(fitted_lambdas_lasso_joined)

fitted_lambdas_lasso_joined_vec_mean <- mean(fitted_lambdas_lasso_joined_vec)

fitted_lambdas_lasso_joined_vec_std <- sd(fitted_lambdas_lasso_joined_vec)

fitted_nonzeros_lasso_joined <- list()

for (i in 1:100){
  
  fitted_nonzeros_lasso_joined[[i]] <- nrow(extract.coef(fitted_lasso_joined[[i]]))
  
} 

fitted_nonzeros_lasso_joined_vec <- unlist(fitted_nonzeros_lasso_joined)

fitted_nonzeros_lasso_joined_vec_mean <- mean(fitted_nonzeros_lasso_joined_vec)

fitted_nonzeros_lasso_joined_vec_std <- sd(fitted_nonzeros_lasso_joined_vec)

################################
####100 JOINED EN25s RESULTS####
################################

MSEs_train_en025_joined_vec <- unlist(MSEs_train_en025_joined)

MSEs_train_en025_joined_vec_mean <- mean(MSEs_train_en025_joined_vec)

MSEs_train_en025_joined_vec_std <- sd(MSEs_train_en025_joined_vec)

MSEs_test_en025_joined_vec <- unlist(MSEs_test_en025_joined)

MSEs_test_en025_joined_vec_mean <- mean(MSEs_test_en025_joined_vec)

MSEs_test_en025_joined_vec_std <- sd(MSEs_test_en025_joined_vec)

fitted_lambdas_en025_joined <- list()

for (i in 1:100){
  
  fitted_lambdas_en025_joined[[i]] <- fitted_en025_joined[[i]]$lambda.min
  
} 

fitted_lambdas_en025_joined_vec <- unlist(fitted_lambdas_en025_joined)

fitted_lambdas_en025_joined_vec_mean <- mean(fitted_lambdas_en025_joined_vec)

fitted_lambdas_en025_joined_vec_std <- sd(fitted_lambdas_en025_joined_vec)

fitted_nonzeros_en025_joined <- list()

for (i in 1:100){
  
  fitted_nonzeros_en025_joined[[i]] <- nrow(extract.coef(fitted_en025_joined[[i]]))
  
} 

fitted_nonzeros_en025_joined_vec <- unlist(fitted_nonzeros_en025_joined)

fitted_nonzeros_en025_joined_vec_mean <- mean(fitted_nonzeros_en025_joined_vec)

fitted_nonzeros_en025_joined_vec_std <- sd(fitted_nonzeros_en025_joined_vec)

################################
####100 JOINED EN50s RESULTS####
################################

MSEs_train_en050_joined_vec <- unlist(MSEs_train_en050_joined)

MSEs_train_en050_joined_vec_mean <- mean(MSEs_train_en050_joined_vec)

MSEs_train_en050_joined_vec_std <- sd(MSEs_train_en050_joined_vec)

MSEs_test_en050_joined_vec <- unlist(MSEs_test_en050_joined)

MSEs_test_en050_joined_vec_mean <- mean(MSEs_test_en050_joined_vec)

MSEs_test_en050_joined_vec_std <- sd(MSEs_test_en050_joined_vec)

fitted_lambdas_en050_joined <- list()

for (i in 1:100){
  
  fitted_lambdas_en050_joined[[i]] <- fitted_en050_joined[[i]]$lambda.min
  
} 

fitted_lambdas_en050_joined_vec <- unlist(fitted_lambdas_en050_joined)

fitted_lambdas_en050_joined_vec_mean <- mean(fitted_lambdas_en050_joined_vec)

fitted_lambdas_en050_joined_vec_std <- sd(fitted_lambdas_en050_joined_vec)

fitted_nonzeros_en050_joined <- list()

for (i in 1:100){
  
  fitted_nonzeros_en050_joined[[i]] <- nrow(extract.coef(fitted_en050_joined[[i]]))
  
} 

fitted_nonzeros_en050_joined_vec <- unlist(fitted_nonzeros_en050_joined)

fitted_nonzeros_en050_joined_vec_mean <- mean(fitted_nonzeros_en050_joined_vec)

fitted_nonzeros_en050_joined_vec_std <- sd(fitted_nonzeros_en050_joined_vec)

################################
####100 JOINED EN75s RESULTS####
################################

MSEs_train_en075_joined_vec <- unlist(MSEs_train_en075_joined)

MSEs_train_en075_joined_vec_mean <- mean(MSEs_train_en075_joined_vec)

MSEs_train_en075_joined_vec_std <- sd(MSEs_train_en075_joined_vec)

MSEs_test_en075_joined_vec <- unlist(MSEs_test_en075_joined)

MSEs_test_en075_joined_vec_mean <- mean(MSEs_test_en075_joined_vec)

MSEs_test_en075_joined_vec_std <- sd(MSEs_test_en075_joined_vec)

fitted_lambdas_en075_joined <- list()

for (i in 1:100){
  
  fitted_lambdas_en075_joined[[i]] <- fitted_en075_joined[[i]]$lambda.min
  
} 

fitted_lambdas_en075_joined_vec <- unlist(fitted_lambdas_en075_joined)

fitted_lambdas_en075_joined_vec_mean <- mean(fitted_lambdas_en075_joined_vec)

fitted_lambdas_en075_joined_vec_std <- sd(fitted_lambdas_en075_joined_vec)

fitted_nonzeros_en075_joined <- list()

for (i in 1:100){
  
  fitted_nonzeros_en075_joined[[i]] <- nrow(extract.coef(fitted_en075_joined[[i]]))
  
} 

fitted_nonzeros_en075_joined_vec <- unlist(fitted_nonzeros_en075_joined)

fitted_nonzeros_en075_joined_vec_mean <- mean(fitted_nonzeros_en075_joined_vec)

fitted_nonzeros_en075_joined_vec_std <- sd(fitted_nonzeros_en075_joined_vec)
