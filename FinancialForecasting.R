# Financial Forecasting

# Import libraries
library(readxl)
library(neuralnet)
library(caret)
library(Metrics)
library(dplyr)

# Load dataset
exchange_rate_data <- read_excel("ExchangeUSD (2).xlsx")
dim(exchange_rate_data)

# Select the USD/EUR exchange rate column (3rd column)
time_series_data <- exchange_rate_data$`USD/EUR`

head(time_series_data)
plot(time_series_data)

# Task B
# Input/output matrices for time-delayed vectors upto (t-4)
lagged_data <- data.frame(T_minus4 = lag(time_series_data,4),
                          T_minus3 = lag(time_series_data,3),
                          T_minus2 = lag(time_series_data,2),
                          T_minus1 = lag(time_series_data,1),
                              T_pred = time_series_data)

# View data
head(lagged_data)

# Remove rows with NA
lagged_data <- na.omit(lagged_data)

head(lagged_data)
str(lagged_data)

# Task C
# Normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to lagged data
lagged_data_normalized <- as.data.frame(lapply(lagged_data, normalize))

# View the normalized data
head(lagged_data_normalized)

# Split train and test data
training_data <- lagged_data_normalized[1:400, ]
testing_data <- lagged_data_normalized[401:nrow(lagged_data_normalized)-4, ]

# View train and test data
head(training_data)
head(testing_data)
plot(training_data)
plot(testing_data)

# Split original train and test data
original_training_data <- time_series_data[1:400] # the first 400 rows
head(original_training_data)
original_testing_data <- time_series_data[401:496] # the remaining rows
head(original_testing_data)

# plot(original_testing_data)

# Get min and max of the original dataset
dataset_min <- min(time_series_data)
dataset_min
dataset_max <- max(time_series_data)
dataset_max

# De-normalization function
denormalize <- function(x, min, max) {
  return( (max - min) * x + min )
}

# Task D
# MLP Models
set.seed(123)

dataset_model1 <- neuralnet(
  formula = T_pred ~ T_minus4 + T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = 12,
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model1)
     
dataset_model2 <- neuralnet(
  formula = T_pred ~ T_minus4 + T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = 8,
  linear.output = FALSE,
  act.fct = 'logistic'
)
plot(dataset_model2)

dataset_model3 <- neuralnet(
  formula = T_pred ~ T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = 6,
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model3)

dataset_model4 <- neuralnet(
  formula = T_pred ~ T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = 4,
  linear.output = FALSE,
  act.fct = 'tanh'
)
plot(dataset_model4)

dataset_model5 <- neuralnet(
  formula = T_pred ~ T_minus2 + T_minus1,
  data = training_data,
  hidden = 12,
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model5)

dataset_model6 <- neuralnet(
  formula = T_pred ~ T_minus2 + T_minus1,
  data = training_data,
  hidden = 8,
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model6)

dataset_model7 <- neuralnet(
  formula = T_pred ~ T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = 12,
  linear.output = TRUE,
  act.fct = 'tanh'
)
plot(dataset_model7)

dataset_model8 <- neuralnet(
  formula = T_pred ~ T_minus4 + T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = c(3,5),
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model8)

dataset_model9 <- neuralnet(
  formula = T_pred ~ T_minus4 + T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = c(2,2),
  linear.output = FALSE,
  act.fct = 'logistic'
)
plot(dataset_model9)

dataset_model10 <- neuralnet(
  formula = T_pred ~ T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = c(2,2),
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model10)

dataset_model11 <- neuralnet(
  formula = T_pred ~ T_minus3 + T_minus2 + T_minus1,
  data = training_data,
  hidden = c(6,8),
  linear.output = FALSE,
  act.fct = 'tanh'
)
plot(dataset_model11)

dataset_model12 <- neuralnet(
  formula = T_pred ~ T_minus1,
  data = training_data,
  hidden = c(5,12),
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model12)

dataset_model13 <- neuralnet(
  formula = T_pred ~ T_minus1,
  data = training_data,
  hidden = c(4,8),
  linear.output = TRUE,
  act.fct = 'logistic'
)
plot(dataset_model13)

dataset_model14 <- neuralnet(
  formula = T_pred ~ T_minus2 + T_minus1,
  data = training_data,
  hidden = c(5,12),
  linear.output = FALSE,
  act.fct = 'logistic',
)
plot(dataset_model14)

dataset_model15 <- neuralnet(
  formula = T_pred ~ T_minus2 + T_minus1,
  data = training_data,
  hidden = c(4,8),
  linear.output = FALSE,
  act.fct = 'tanh'
)
plot(dataset_model15)

# Predict rates for each model
predicted_results_model1 <- predict(dataset_model1, testing_data)
predicted_results_model2 <- predict(dataset_model2, testing_data)
predicted_results_model3 <- predict(dataset_model3, testing_data)
predicted_results_model4 <- predict(dataset_model4, testing_data)
predicted_results_model5 <- predict(dataset_model5, testing_data)
predicted_results_model6 <- predict(dataset_model6, testing_data)
predicted_results_model7 <- predict(dataset_model7, testing_data)
predicted_results_model8 <- predict(dataset_model8, testing_data)
predicted_results_model9 <- predict(dataset_model9, testing_data)
predicted_results_model10 <- predict(dataset_model10, testing_data)
predicted_results_model11 <- predict(dataset_model11, testing_data)
predicted_results_model12 <- predict(dataset_model12, testing_data)
predicted_results_model13 <- predict(dataset_model13, testing_data)
predicted_results_model14 <- predict(dataset_model14, testing_data)
predicted_results_model15 <- predict(dataset_model15, testing_data)

# View data of prediction
head(predicted_results_model8)

# Get target and view data
predicted_T <- predicted_results_model8
head(predicted_T)
dim(predicted_T)

# De-normalize the predicted output
target_prediction <- denormalize(predicted_T, dataset_min, dataset_max)
head(target_prediction) 
dim(target_prediction)
#plot(target_prediction)

# Standard statistical indices
rmse_val <- rmse(unlist(original_testing_data), unlist(target_prediction))
rmse_val

mae_val <- mae(unlist(original_testing_data), unlist(target_prediction))
mae_val

mape_val <- mape(unlist(original_testing_data), unlist(target_prediction))
mape_val

smape_val <- smape(unlist(original_testing_data), unlist(target_prediction))
smape_val

# Task G
# After selection of best model
selected_best_model <- dataset_model8

# Weight parameter calculation
num_of_inputs <- 4
num_of_hidden_layer <- 2
num_of_hidden_nodes_l1 <- 3
num_of_hidden_nodes_l2 <- 5

weight_parameters <- (num_of_inputs * num_of_hidden_nodes_l1) + (num_of_hidden_nodes_l1 * num_of_hidden_nodes_l2)
weight_parameters

# Task H
# Scatter Plot
# Plot predicted output vs. desired output
plot(unlist(original_testing_data), unlist(target_prediction),
     xlab = "Desired Output", ylab = "Predicted Output",
     main = "Predicted vs. Desired Output")
abline(0, 1, col = "red")  

# Line Chart
# x-axis' time indices
time_indices <- seq_along(unlist(original_testing_data))

# Plot original testing data
plot(time_indices, unlist(original_testing_data), type = "l", col = "blue",
     xlab = "Time", ylab = "Exchange Rate",
     main = "Predicted vs. Desired Output")

# Predicted output
lines(time_indices, unlist(target_prediction), col = "red")

# Legend
legend("topleft", legend = c("Desired Output", "Predicted Output"),
       col = c("blue", "red"), lty = 1)



