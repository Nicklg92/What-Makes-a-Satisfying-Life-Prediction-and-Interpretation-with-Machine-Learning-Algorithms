#################################################################
##SEVENTH SCRIPT: RIDGE COEFFS TO COMPARE WITH SVs, EXTENDED SET#
#################################################################

library(glmnet)
library(coefplot)

options(scipen=999)

rm(list = ls())

set.seed(42)

#COMMENTS:

#This is the seventh script in the sequence necessary to produce
#the results in "What Makes a Satisfying Life? Prediction and 
#Interpretation with Machine-Learning Algorithms", 
#Gentile et al. (2022). 

#In particular, this is the second of those necessary
#to produce the regression coefficients to be compared
#with the Shapley Values.

#For the Original model, we used the coefficients
#from the non-penalized linear regression, whereas for
#the Extended model, as specified in the paper, we
#use the coefficients from a (penalized) ridge
#regression.

#The reason for this is that if we were to consider 
#the coefficients from the non-penalized model,
#we would not have a coefficients for all the dropped
#dummies, which would therefore deflate the value
#of the derived coefficient (equations 15 and 16) in 
#the paper.

import_path <- "C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_test_splits\\"

train_1 <- read.csv(paste0(import_path, "train_1.csv"))

test_1 <- read.csv(paste0(import_path, "test_1.csv"))

X_train_1 <- train_1[,-which(names(train_1) %in% c("Educational.Achievement", "Employed",
                                                   "Has.a.Partner", "lifesatisfaction", "Any.long.standing.illness"))]

y_train_1 <- train_1["lifesatisfaction"]

X_test_1 <- test_1[,-which(names(test_1) %in% c("Educational.Achievement", "Employed",
                                                   "Has.a.Partner", "lifesatisfaction", "Any.long.standing.illness"))]

y_test_1 <- test_1["lifesatisfaction"]

cvfit_ridge_joined <- cv.glmnet(x = as.matrix(X_train_1),
                                y = as.matrix(y_train_1),
                                standardize = TRUE,
                                nfolds = 5,
                                alpha = 0)

fitted_test_ridge_joined <- predict(cvfit_ridge_joined, newx = as.matrix(X_test_1), s = "lambda.min")

MSE_test_ridge_joined <- colMeans((y_test_1 - fitted_test_ridge_joined)^(2))

fitted_training_ridge_joined <- predict(cvfit_ridge_joined, newx = as.matrix(X_train_1), s = "lambda.min")

MSE_train_ridge_joined <- colMeans((y_train_1 - fitted_training_ridge_joined)^(2))

ridge_coeffs <- as.matrix(coef(cvfit_ridge_joined, s = "lambda.min"))

write.csv(ridge_coeffs, file ="C:\\Users\\niccolo.gentile\\Desktop\\BCS\\Train_1_joined_ridge_coeffs_all.csv", row.names = TRUE)

