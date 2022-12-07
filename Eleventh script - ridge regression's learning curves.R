########################################################
##ELEVENTH SCRIPT _ RIDGE REGRESSION'S LEARNING CURVES##
########################################################

library(glmnet)
library(coefplot)

options(scipen=999)

#COMMENTS:

#This is the eleventh script in the sequence necessary to produce
#the results in "What Makes a Satisfying Life? Prediction and 
#Interpretation with Machine-Learning Algorithms", 
#Gentile et al. (2022). 

#In particular, here we create the data for Figure 6, 
#the Learning Curve for the Ridge Regression in the
#Extended model.

#The operations are similar to those done for the
#Learning Curves on the Non-Penalized Linear Regression
#in the Original model.

import_path <- "C:\\some_local_path\\"

train_1 <- read.csv(paste0(import_path, "train_1.csv"))

test_1 <- read.csv(paste0(import_path, "test_1.csv"))

names(train_1) <- gsub(x = names(train_1), pattern = "\\.", replacement = " ")

X_train_1 <- train_1[,-which(names(train_1) %in% c('Educational.Achievement', 'Employed',
                                                   'Has.a.Partner', 'lifesatisfaction', "Any.long.standing.illness"))]

y_train_1 <- train_1["lifesatisfaction"]

names(test_1) <- gsub(x = names(test_1), pattern = "\\.", replacement = " ")

X_test_1 <- test_1[,-which(names(test_1) %in% c('Educational.Achievement', 'Employed',
                                                'Has.a.Partner', 'lifesatisfaction', "Any.long.standing.illness"))]

y_test_1 <- test_1["lifesatisfaction"]

train_sizes <- c(0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06,
                 0.05, 0.04, 0.03, 0.02, 0.01)

X <- rbind(X_train_1, X_test_1)

y <- rbind(y_train_1, y_test_1)

MSEs_test_ridge <- list()

train_size_list <- list()

for (i in train_sizes){
  
  seed <- sample.int(1000, 1)   
  
  train_size <- floor(i * nrow(X))
  
  train_ind <- sample(seq_len(nrow(X)), size = train_size)
  
  X_train <- X[train_ind, ]
  
  X_test <- X[-train_ind, ]
  
  y_train <- y[train_ind, ]
  
  y_test <- y[-train_ind, ]
  
  index <- match(i, train_sizes)
  
  cvfit_ridge_joined <- cv.glmnet(x = as.matrix(X_train),
                                  y = as.matrix(y_train),
                                  standardize = TRUE,
                                  nfolds = 5,
                                  alpha = 0)
  
  
  fitted_test_ridge_joined <- predict(cvfit_ridge_joined, newx = as.matrix(X_test), s = "lambda.min")
  
  MSE_test_ridge_joined <- colMeans((y_test - fitted_test_ridge_joined)^(2))
  
  MSEs_test_ridge[[index]] <- MSE_test_ridge_joined
  
  train_size_list[[index]] <- i * 100
  
}  

MSEs_and_sizes <- data.frame(unlist(MSEs_test_ridge), unlist(train_size_list))

names(MSEs_and_sizes) <- c("Test MSEs", "Train size (%)")

write.csv(MSEs_and_sizes, file = "C:\\some_other_local_path\\Ridge_extd_learning_curve.csv")
