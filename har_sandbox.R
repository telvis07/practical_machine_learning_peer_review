# R script with exploratory scripts for the Human Activity Recording data
# http://groupware.les.inf.puc-rio.br/har

# Report Guidelines
# 1. You may use any of the other variables to predict with. 
# 2. You should create a report describing how you built your model, 
# 3. how you used cross validation, 
# 4. what you think the expected out of sample error is, 
# 5. and why you made the choices you did. 
# 6. You will also use your prediction model to predict 20 different test cases.

# Submission Guidelines
# Link to a Github repo with your R markdown and compiled HTML file describing your analysis. 
# Please constrain the text of the writeup to < 2000 words and 
# the number of figures to be less than 5. 
# submit a repo with a gh-pages branch so the HTML page can be viewed online.
library(dplyr)
library(caret)
library(rpart)

# df <- load_training_datums()
# vi <- feature_selection_rf(df)
# df_imputed <- impute_training_rf(df)
# modFit <- train_rf_all_features(df_imputed)
# modFit_vi <- train_rf_top_features(df_imputed, vi)


load_training_datums <- function() {
  training <- read.csv("data/pml-training.csv", 
                       na.strings = c("NA", "", "#DIV/0!"), 
                       strip.white = TRUE, 
                       stringsAsFactors = FALSE)
  
  # data cols
  allcols <- names(training)
  train_measure_cols <- allcols[grep("belt|arm|dumbbell",allcols)]
  train_label_cols <- allcols[-grep("belt|arm|dumbbell",allcols)]
  
  # get only the columns with sensor data
  df <- subset(training, select=train_measure_cols)
  
  # find columns where all the sensor data is null.
  all_null_cols <- train_measure_cols[apply(df, 2, function(x) sum(is.na(x)) == length(x))]
  train_measure_cols_not_null <- setdiff(train_measure_cols, all_null_cols)
  
  # get only measurement columns with non-null values
  df <- subset(training, select=train_measure_cols_not_null)
  
  #
  df$classe <- as.factor(training$classe)
 
  # summary
  total_variables_in_training_set <- length(names(training))
  total_variables_with_belt_arm_dumbell_data <- length(train_measure_cols)
  total_variables_with_nonnull_belt_arm_dumbell_data <- length(train_measure_cols_not_null)
  total_rows_with_all_data <- sum(apply(df, 1, function(x) sum(is.na(x)) == 0))
  print(sprintf("Number of Variables in Training Data: %s", total_variables_in_training_set))
  print(sprintf("Number of Belt, Arm, Dumbell Sensor Variables: %s", total_variables_with_belt_arm_dumbell_data))
  print(sprintf("Number of Belt, Arm, Dumbell Variables with non-NA values: %s", total_variables_with_nonnull_belt_arm_dumbell_data))
  print(sprintf("Number of outcome variables (classe): %s", 1))
  print(sprintf("Number of rows where a variables are non-NA %s", total_rows_with_all_data ))
  
  # [1] "Number of Variables in Training Data: 160"
  # [1] "Number of Belt, Arm, Dumbell Sensor Variables: 152"
  # [1] "Number of Belt, Arm, Dumbell Variables with non-NA values: 146"
  # [1] "Number of outcome variables (classe): 1"
  
  print("Saving RDS")
  saveRDS(df, "data/datums.rds")
  
  df
  # testing <- read.csv("data/pml-testing.csv", na.strings = "NA", stringsAsFactors = FALSE)
  # # data cols
  # allcols <- names(testing)
  # test_measure_cols <- allcols[grep("belt|arm|dumbbell",allcols)]
  # test_label_cols <- allcols[-grep("belt|arm|dumbbell",allcols)]
  # print(test_measure_cols)
  # print(test_label_cols)
}

feature_selection_rf <- function(df) {
  set.seed(33833)
  print("Training Random Forest")
  modFit <- train(classe ~ ., data=df, method="rf", prox=TRUE)
  print("Saving RDS")
  saveRDS(modFit, "data/rf_fit_variable_importance.rds")
  
  # find most important features
  vi <- varImp(modFit, scale=FALSE)$importance
  vi <- data.frame(varname=row.names(vi), Overall=vi$Overall)
  vi <- vi[order(vi$Overall, decreasing = TRUE),]
  
  # varnames with gini greater than 2.
  rf_important_varnames <- vi[vi$Overall > 1,]$varname
  cnt <- length(rf_important_varnames)
  sprintf("Number of Random Forest Features with Gini > 2.0: %s", rf_important_varnames)
  print(head(vi, 20))
  
  # plot the top features
  varImpPlot(modFit$finalModel, 
             scale=FALSE, 
             n.var=length(rf_important_varnames),
             main=sprintf("Top %s RF Variables Ranked by Gini", length(rf_important_varnames)))
  # print("Saving RDS")
  # saveRDS(vi, "data/rf_variable_importance_df.rds")
  
  # return the most important features
  vi

  # 93     avg_roll_dumbbell 11.260981
  # 20      stddev_roll_belt 10.639381
  # 21         var_roll_belt 10.400259
  # 92    var_accel_dumbbell  8.436795
  # 121     min_roll_forearm  8.010433
  # 22        avg_pitch_belt  5.201425
  # 132    avg_pitch_forearm  4.918795
  # 2             pitch_belt  4.864473
  # 129     avg_roll_forearm  4.370307
  # 118     max_roll_forearm  4.057306
  # 106     accel_dumbbell_y  3.949713
  # 109    magnet_dumbbell_y  3.816680
  # 73   amplitude_pitch_arm  2.391769
  # 19         avg_roll_belt  2.353988
  # 18  var_total_accel_belt  2.266188
  # 75         roll_dumbbell  2.244240
  # 110    magnet_dumbbell_z  2.174386
  # 3               yaw_belt  2.139424
  # 112        pitch_forearm  1.940954
  # 143      accel_forearm_z  1.661493
}


rf_vi_stats <- function() {
  vi <- readRDS("data/rf_variable_importance_df.rds")
  
  # vi is a sorted variable importance matrix from varImp() 
  length(rf_important_varnames)
  sum(grepl('dumbbell', rf_important_varnames))
  rf_important_varnames[(grepl('dumbbell', rf_important_varnames))]
  
  sum(grepl('_arm', rf_important_varnames))
  rf_important_varnames[(grepl('_arm', rf_important_varnames))]
  
  sum(grepl('_forearm', rf_important_varnames))
  rf_important_varnames[(grepl('_forearm', rf_important_varnames))]
  
  sum(grepl('belt', rf_important_varnames))
  rf_important_varnames[(grepl('belt', rf_important_varnames))]
  
  # > sum(grepl('dumbbell', rf_important_varnames))
  # [1] 6
  # > rf_important_varnames[(grepl('dumbbell', rf_important_varnames))]
  # [1] avg_roll_dumbbell  var_accel_dumbbell accel_dumbbell_y   magnet_dumbbell_y  roll_dumbbell     
  # [6] magnet_dumbbell_z 
  # > 
  #   > sum(grepl('_arm', rf_important_varnames))
  # [1] 1
  # > rf_important_varnames[(grepl('_arm', rf_important_varnames))]
  # [1] amplitude_pitch_arm
  # > 
  #   > sum(grepl('_forearm', rf_important_varnames))
  # [1] 4
  # > rf_important_varnames[(grepl('_forearm', rf_important_varnames))]
  # [1] min_roll_forearm  avg_pitch_forearm avg_roll_forearm  max_roll_forearm 
  # > 
  #   > sum(grepl('belt', rf_important_varnames))
  # [1] 7
  # > rf_important_varnames[(grepl('belt', rf_important_varnames))]
  # [1] stddev_roll_belt     var_roll_belt        avg_pitch_belt       pitch_belt           var_total_accel_belt
  # [6] yaw_belt             avg_roll_belt       

  # Curl in five different fash- ions:
  # exactly according to the specification (Class A), 
  # throw- ing the elbows to the front (Class B), 
  # lifting the dumbbell only halfway (Class C), 
  # lowering the dumbbell only halfway (Class D) 
  # and throwing the hips to the front (Class E).
}

impute_training_rf <- function(df) {
  # # first impute the missing data
  # print("Start impute")
  df_imputed <- rfImpute(classe ~ ., df, iter=5, ntree=10)
  
  print("Saving RDS")
  saveRDS(df_imputed, "data/rf_imputed_training_df.rds")
  df_imputed
  
  # > sum(apply(df_imputed, 1, function (x) any(is.na(x))))
  # [1] 0
  # > sum(apply(df_imputed, 2, function (x) any(is.na(x))))
  # [1] 0
  
  ## Missing Values : Only 217 or 19622 rows in the dataset have all values for
  # the measurements (i.e. no NA's)
  # > sum(apply(df, 1, function(x) sum(is.na(x)) == 0))
  # [1] 217
  # > sum(apply(testing, 1, function(x) sum(is.na(x)) == 0))
  # [1] 61
  # > length(prediction_rf)
  # [1] 61
  # > sum(apply(training, 1, function(x) sum(is.na(x)) == 0))
  # [1] 156
  # > preds <- predict(modFit, training)
  # > length(preds)
  # [1] 156
}

train_rf_all_features <- function(df_imputed) {
  set.seed(1234)
  inTrain = createDataPartition(df_imputed$classe, p = 3/4)[[1]]
  training = df_imputed[ inTrain,]
  testing = df_imputed[-inTrain,] 
  
  # train with original 147 features
  print("Train fit with 147 features")
  modFit <- train(classe ~ ., 
                  data=training, 
                  method="rf", ntree=100,
                  trControl = trainControl(method="cv"), number=10)

  # predictions
  df_imputed
  
  # testing
  prediction_rf <- predict(modFit, testing)
  print("Predictions RF with 147 features: testing")
  print(confusionMatrix(prediction_rf, testing$classe))
  
  # training
  prediction_rf <- predict(modFit, training)
  print("Predictions RF with 147 features:training")
  print(confusionMatrix(prediction_rf, training$classe))
  
  print("Saving RDS")
  saveRDS(modFit, "data/rf_fit_all_features.rds")
  
  # return model with all features
  modFit
}

train_rf_top_features <- function(df_imputed, vi, gini_threshold=1) {
  set.seed(1234)
  inTrain = createDataPartition(df_imputed$classe, p = 3/4)[[1]]
  training = df_imputed[ inTrain,]
  testing = df_imputed[-inTrain,] 

  
  # modfit with most important features  
  rf_important_varnames <- vi[vi$Overall > gini_threshold,]$varname
  sprintf("Number of Random Forest Features with Gini > %s: %s", gini_threshold, rf_important_varnames)
  print(head(vi, length(rf_important_varnames)))
  
  reduced_training <- subset(training, select=c(as.vector(rf_important_varnames), "classe"))
  reduced_testing <- subset(testing, select=c(as.vector(rf_important_varnames), "classe"))
  
  print("Train fit with 17 features")
  modFit_vi <- train(classe ~ ., 
                     data=reduced_training, 
                     method="rf", ntree=100, 
                     trControl = trainControl(method="cv"), number=10)
  # testing
  prediction_rf_vi <- predict(modFit_vi, reduced_testing)
  print("Predictions RF with top features: testing")
  print(confusionMatrix(prediction_rf_vi, reduced_testing$classe))
  
  # training
  prediction_rf_vi <- predict(modFit_vi, reduced_training)
  print("Predictions RF with top features: training")
  print(confusionMatrix(prediction_rf_vi, reduced_training$classe))
  
  
  print("Saving RDS")
  saveRDS(modFit_vi, sprintf("data/rf_fit_%s_features.rds", length(rf_important_varnames)))
  
  # return model with top features
  modFit_vi
}


plot_confusion_reduced <- function(){
  set.seed(1234)
  
  # read datums
  vi <- readRDS("data/rf_variable_importance_df.rds")
  modFit_vi <- readRDS("data/rf_fit_36_features.rds")
  df_imputed <- readRDS("data/rf_imputed_training_df.rds")
  gini_threshold = 1

  # split
  inTrain = createDataPartition(df_imputed$classe, p = 3/4)[[1]]
  training = df_imputed[ inTrain,]
  testing = df_imputed[-inTrain,] 
  
  # modfit with most important features  
  rf_important_varnames <- vi[vi$Overall > gini_threshold,]$varname
  sprintf("Number of Random Forest Features with Gini > %s: %s", gini_threshold, rf_important_varnames)
  print(head(vi, length(rf_important_varnames)))
  
  reduced_training <- subset(training, select=c(as.vector(rf_important_varnames), "classe"))
  reduced_testing <- subset(testing, select=c(as.vector(rf_important_varnames), "classe"))
  
  # testing
  prediction_rf_vi <- predict(modFit_vi, reduced_testing)
  print("Predictions RF with top features: testing")
  confuse <- confusionMatrix(prediction_rf_vi, reduced_testing$classe)
  confuse.percent <- apply(confuse$table, 1, function(x) x/sum(x))
  confuse.melt <- melt(confuse.percent)
  confuse.df <- as.data.frame(confuse.melt)
  
  ggplot(confuse.df, aes(x=Reference, y=Prediction, fill=value)) +
    geom_tile(aes(fill=value)) +
    geom_text(aes(label = round(value, 3))) +
    scale_fill_gradient(low="white",high="red") +
    ggtitle("Normalized Confusion Matrix for \n Random forest with 34 predictors") +
    xlab("Actual Class") +
    ylab("Predicted Class") 
}


plot_confusion_all <- function(){
  set.seed(1234)
  
  # read datums
  modFit <- readRDS("data/rf_fit_all_features.rds")
  df_imputed <- readRDS("data/rf_imputed_training_df.rds")

  # split
  inTrain = createDataPartition(df_imputed$classe, p = 3/4)[[1]]
  training = df_imputed[ inTrain,]
  testing = df_imputed[-inTrain,] 
  
  
  # testing
  prediction_rf <- predict(modFit, testing)
  print("Predictions RF with top features: testing")
  confuse <- confusionMatrix(prediction_rf, testing$classe)
  confuse.percent <- apply(confuse$table, 1, function(x) x/sum(x))
  confuse.melt <- melt(confuse.percent)
  confuse.df <- as.data.frame(confuse.melt)
  
  ggplot(confuse.df, aes(x=Reference, y=Prediction, fill=value)) +
    geom_tile(aes(fill=value)) +
    geom_text(aes(label = round(value, 3))) +
    scale_fill_gradient(low="white",high="red") +
    ggtitle("Normalized Confusion Matrix for \n Random forest with 156 predictors") +
    xlab("Actual Class") +
    ylab("Predicted Class") 
}


predict_validation_set <- function(){
  # pml_testing <- read.csv("data/pml-testing.csv", 
  #                         na.strings = c("NA", "", "#DIV/0!"), 
  #                         strip.white = TRUE, 
  #                         stringsAsFactors = FALSE)
  
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
}

plot_feature_importance <- function() {
    set.seed(33833)

    modFit <- readRDS("data/rf_fit_variable_importance.rds")
    
    # plot the top features
    varImpPlot(modFit$finalModel, 
               scale=FALSE, 
               n.var=length(rf_important_varnames),
               main=sprintf("Top %s RF Variables Ranked by Gini", length(rf_important_varnames)))
}

data_summary_stats <- function() {
  training <- readRDS("data/pml_training_csv.rds")
  
  # data cols
  allcols <- names(training)
  train_measure_cols <- allcols[grep("belt|arm|dumbbell",allcols)]
  train_label_cols <- allcols[-grep("belt|arm|dumbbell",allcols)]
  
  # get only the columns with sensor data
  df <- subset(training, select=train_measure_cols)
  
  # find columns where all the sensor data is null.
  all_null_cols <- train_measure_cols[apply(df, 2, function(x) sum(is.na(x)) == length(x))]
  train_measure_cols_not_null <- setdiff(train_measure_cols, all_null_cols)
  
  # get only measurement columns with non-null values
  df <- subset(training, select=train_measure_cols_not_null)
  
  #
  df$classe <- as.factor(training$classe)
  
  # summary
  total_variables_in_training_set <- length(names(training))
  total_variables_with_belt_arm_dumbell_data <- length(train_measure_cols)
  total_variables_with_nonnull_belt_arm_dumbell_data <- length(train_measure_cols_not_null)
  total_rows_with_all_data <- sum(apply(df, 1, function(x) sum(is.na(x)) == 0))
  print(sprintf("Number of Variables in Training Data: %s", total_variables_in_training_set))
  print(sprintf("Number of Belt, Arm, Dumbell Sensor Variables: %s", total_variables_with_belt_arm_dumbell_data))
  print(sprintf("Number of Belt, Arm, Dumbell Variables with non-NA values: %s", total_variables_with_nonnull_belt_arm_dumbell_data))
  print(sprintf("Number of outcome variables (classe): %s", 1))
  print(sprintf("Number of rows where a variables are non-NA: %s", total_rows_with_all_data ))
  
  # [1] "Number of Variables in Training Data: 160"
  # [1] "Number of Belt, Arm, Dumbell Sensor Variables: 152"
  # [1] "Number of Belt, Arm, Dumbell Variables with non-NA values: 146"
  # [1] "Number of outcome variables (classe): 1"
  # [1] "Number of rows where a variables are non-NA 217"
  library(knitr)
  df <- data.frame(total_variables_in_training_set=total_variables_in_training_set, 
                   total_variables_with_belt_arm_dumbell_data=total_variables_with_belt_arm_dumbell_data, 
                   total_variables_with_nonnull_belt_arm_dumbell_data= total_variables_with_nonnull_belt_arm_dumbell_data,
                   num_outcome_variables =1,
                   total_rows_with_all_data=total_rows_with_all_data)
  rnames <- c("total_variables_in_training_set", 
              "total_variables_with_belt_arm_dumbell_data", 
              "total_variables_with_nonnull_belt_arm_dumbell_data",
              "num outcome variables",
              "total_rows_with_all_data")

  kable(df, format = "html")
  NA
}






