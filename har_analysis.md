# Prediction of exercise quality using body sensor data
Telvis Calhoun  
March 17, 2016  

## Executive Summary

TODO:

## Data Preparation
First, lets load libraries and datasets used in the analysis.


```r
library(knitr)
library(dplyr)
library(randomForest)
library(caret)
library(reshape2)
library(grid)
library(gridExtra)
```


First we remove columns that have all NAs. Next we use [rfImpute](http://www.inside-r.org/packages/cran/randomforest/docs/rfImpute) to fill in NA values.


                                              value
-------------------------------------------  ------
total_variables_with_belt_arm_dumbell_data      152
number_variables_all_missing                      6
total_rows                                    19622
total_rows_with_missing_data                  19405

## Feature Selection
We calculate feature importance the [varImp function](http://www.inside-r.org/packages/cran/randomforest/docs/importance) provided by the randomForest package.

![](har_analysis_files/figure-html/unnamed-chunk-3-1.png) 

## Model Selection


Random Forest model with 100 trees. We tried models with all 156 features and 36 features returned found in feature selection. We split the training in data in to 75% training and 25% testing.


```r
set.seed(1234)
  
# read datums
modFit <- readRDS("data/rf_fit_all_features.rds")
df_imputed <- readRDS("data/rf_imputed_training_df.rds")

# split
inTrain = createDataPartition(df_imputed$classe, p = 3/4)[[1]]
training = df_imputed[ inTrain,]
testing = df_imputed[-inTrain,] 
```

The figure below show the confusion matrices for models built with all 156 predictors and the best 34 prediction with GINI greater than 1.0. The accuracy for model with all features is TODO. However the model with the best 36 features performs just as well.

<img src="har_analysis_files/figure-html/unnamed-chunk-5-1.png" title="" alt="" style="display: block; margin: auto;" />

## Prediction with Test Data

The test data contains several columns with missing data. We use the cleaned data to impute the missing values using [na.roughfix](http://www.inside-r.org/packages/cran/randomforest/docs/na.roughfix). 

TODO: Put stats here about the missing data in pml-testing.


Table: pml_testing_csv summary

                                       value
------------------------------------  ------
total_columns                            160
total_columns_with_all_missing_data      100
total_rows                                20
total_rows_with_missing_data              20

## Conclusion

TODO:

# Appendix

## Predictions for pml_testing data


```r
pml_testing <- readRDS("data/pml_testing_csv.rds")
  
# read datums
vi <- readRDS("data/rf_variable_importance_df.rds")
modFit_vi <- readRDS("data/rf_fit_36_features.rds")
df_imputed <- readRDS("data/rf_imputed_training_df.rds")
gini_threshold = 1

# data cols
allcols <- names(pml_testing)
train_measure_cols <- allcols[grep("belt|arm|dumbbell",allcols)]
rf_important_varnames <- vi[vi$Overall > gini_threshold,]$varname

# pml-testing data for 'most important' predictors
reduced_pml_testing <- subset(pml_testing, select=train_measure_cols)
reduced_pml_testing <- subset(pml_testing, select=c(as.vector(rf_important_varnames)))
reduced_pml_testing$problem_id <- pml_testing$problem_id


# training data for 'most important' predictors
reduced_df_imputed <- subset(df_imputed, select=c(as.vector(rf_important_varnames)))
reduced_df_imputed$classe <- df_imputed$classe


###########################################
# replace NA's in the test data.
# simply replace with the median for the column
# For numeric variables, NAs are replaced with column medians
reduced_pml_testing$is_test_data <- TRUE
reduced_df_imputed$is_test_data <- FALSE

# combine the data frames, remove 'classe' and 'problem_id' cols
combined_data <- rbind(subset(reduced_df_imputed, select=-c(classe)), subset(reduced_pml_testing, select=-c(problem_id)))
combined_rough <- na.roughfix(combined_data[,-length(combined_data)])
# get the roughfix data
combined_rough$is_test_data <- combined_data$is_test_data
pml_imputed <- filter(combined_rough, is_test_data==TRUE)

# predict on imputed pml-testing data
predict(modFit_vi, pml_imputed)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


# Citations

[1] Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
Cited by 2 (Google Scholar)

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz439hx3Sdf

