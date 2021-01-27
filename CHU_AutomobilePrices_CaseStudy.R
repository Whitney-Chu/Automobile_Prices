## ----setup, include=FALSE-----------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## -----------------------------------------------------------------------------------------------------------
library(datasets)
library(neuralnet)
library(caret)
library(reshape2)
library(ggplot2)
library(stringr)
library(nnet)
library(e1071)
library(DMwR)
library(mltools)


## -----------------------------------------------------------------------------------------------------------
auto.data<- read.csv("AutomobilePrices.csv")


## -----------------------------------------------------------------------------------------------------------
auto.data
symboling <- auto.data$symboling
b <- auto.data[,4:26]
auto.data <- cbind(symboling, b)
auto.data
head(auto.data)
summary(auto.data)
dim(auto.data)
str(auto.data)


## -----------------------------------------------------------------------------------------------------------
colSums(is.na(auto.data))


## -----------------------------------------------------------------------------------------------------------
auto.data$fueltype <- as.factor(auto.data$fueltype)
auto.data$aspiration <- as.factor(auto.data$aspiration)
auto.data$doornumber <- as.factor(auto.data$doornumber)
auto.data$carbody <- as.factor(auto.data$carbody)
auto.data$drivewheel <- as.factor(auto.data$drivewheel)
auto.data$enginelocation <- as.factor(auto.data$enginelocation)
auto.data$enginetype <- as.factor(auto.data$enginetype)
auto.data$cylindernumber <- as.factor(auto.data$cylindernumber)
auto.data$fuelsystem <- as.factor(auto.data$fuelsystem)
str(auto.data)


## -----------------------------------------------------------------------------------------------------------
mod <-  lm(price ~ enginesize  +  carbody +  enginelocation + carwidth + curbweight + enginetype +  enginesize + fuelsystem +  stroke  + peakrpm, 	data = auto.data)
summary(mod)
plot(mod)


## -----------------------------------------------------------------------------------------------------------
auto.data$fueltype <- as.numeric(auto.data$fueltype)
auto.data$aspiration <- as.numeric(auto.data$aspiration)
auto.data$doornumber <- as.numeric(auto.data$doornumber)
auto.data$carbody <- as.numeric(auto.data$carbody)
auto.data$drivewheel <- as.numeric(auto.data$drivewheel)
auto.data$enginelocation <- as.numeric(auto.data$enginelocation)
auto.data$enginetype <- as.numeric(auto.data$enginetype)
auto.data$cylindernumber <- as.numeric(auto.data$cylindernumber)
auto.data$fuelsystem <- as.numeric(auto.data$fuelsystem)
str(auto.data)


## -----------------------------------------------------------------------------------------------------------
set.seed(100)
rows <- c(1:nrow(auto.data))
split <- sample(rows,size = (nrow(auto.data)*0.75))

train1 <- as.data.frame(auto.data[split,])
train.scale <- scale(train1)
train <- as.data.frame(train.scale)

test1 <- as.data.frame(auto.data[-split,])
test.scale <- scale(test1)
test <- as.data.frame(test.scale)

nrow(train)
head(train)
nrow(test)
head(test)
auto.data


## -----------------------------------------------------------------------------------------------------------
nn <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody + 
    drivewheel + enginelocation + wheelbase + carlength + carwidth + 
    carheight + curbweight + enginetype + cylindernumber + enginesize + 
    fuelsystem + boreratio + stroke + compressionratio + horsepower + 
    peakrpm + citympg + highwaympg
, data = train, hidden = c(12,6))
nn$result.matrix
plot(nn)


## -----------------------------------------------------------------------------------------------------------
pred1 <- compute(nn, train[,1:23])
results.train1 <- data.frame(actual = train$price, prediction = pred1$net.result)
results.train1
RMSE(pred1$net.result, train$price)


## -----------------------------------------------------------------------------------------------------------
pred2 <- compute(nn, test[,1:23])
results.test1 <- data.frame(actual = test$price, prediction = pred2$net.result)
results.test1
RMSE(pred2$net.result, test$price)


## -----------------------------------------------------------------------------------------------------------
unscaled1 <- (pred1$net.result) * sd(train1$price) + mean(train1$price)
unscaled1
results.train2 <- data.frame(actual = train1$price, prediction = unscaled1)
results.train2


## -----------------------------------------------------------------------------------------------------------
unscaled2 <- (pred2$net.result) * sd(test1$price) + mean(test1$price)
unscaled2
results.test2 <- data.frame(actual = test1$price, prediction = unscaled2)
results.test2


## -----------------------------------------------------------------------------------------------------------
plot(results.train2, col='blue', pch=16, main = "predicted vs actual for the training set", ylab = "predicted", xlab = "actual")
plot(results.test2, col='blue', pch=16, main = "predicted vs actual for the testing set", ylab = "predicted", xlab = "actual")


## -----------------------------------------------------------------------------------------------------------
set.seed(100)
train.cont <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
mod.lm <-  train(price ~ enginesize  +  carbody +  enginelocation + carwidth + curbweight + enginetype +  enginesize + fuelsystem +  stroke  + peakrpm, 	data = train)
print(mod.lm)


## -----------------------------------------------------------------------------------------------------------
folds <- createFolds(train$price, k = 10)
str(folds)
results <- c()
for (fld in folds){
  index <- sample(1:nrow(train),round(0.9*nrow(train)))
  data <- train[-fld,]
  nn <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody +
    drivewheel + enginelocation + wheelbase + carlength + carwidth +
    carheight + curbweight + enginetype + cylindernumber + enginesize +
    fuelsystem + boreratio + stroke + compressionratio + horsepower +
    peakrpm + citympg + highwaympg, data = train, hidden = c(4,2))
  pred.val1 <- compute(nn, train[,1:23])
  results <- cbind(results,RMSE(pred.val1$net.result, train$price))
}
paste("After", length(results), "validation loops the mean error of the network is", paste0(round(mean(results),2), "%"))


## -----------------------------------------------------------------------------------------------------------
folds <- createFolds(test$price, k = 10)
str(folds)
results <- c()
for (fld in folds){
  index <- sample(1:nrow(test),round(0.9*nrow(test)))
  data <- test[-fld,]
  nn <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody +
    drivewheel + enginelocation + wheelbase + carlength + carwidth +
    carheight + curbweight + enginetype + cylindernumber + enginesize +
    fuelsystem + boreratio + stroke + compressionratio + horsepower +
    peakrpm + citympg + highwaympg, data = test, hidden = c(4,2))
  pred.val2 <- compute(nn, test[,1:23])
  results <- cbind(results,RMSE(pred.val2$net.result, test$price))
}
paste("After", length(results), "validation loops the mean error of the network is", paste0(round(mean(results),2), "%"))


## -----------------------------------------------------------------------------------------------------------
nn2 <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody + 
    drivewheel + enginelocation + wheelbase + carlength + carwidth + 
    carheight + curbweight + enginetype + cylindernumber + enginesize + 
    fuelsystem + boreratio + stroke + compressionratio + horsepower + 
    peakrpm + citympg + highwaympg
, data = train, hidden = c(6,6))
nn2$result.matrix
plot(nn2)


## -----------------------------------------------------------------------------------------------------------
pred3 <- compute(nn2, train[,1:23])
results.train2 <- data.frame(actual = train$price, prediction = pred3$net.result)
results.train2
RMSE(pred3$net.result, train$price)


## -----------------------------------------------------------------------------------------------------------
pred4 <- compute(nn2, test[,1:23])
results.test2 <- data.frame(actual = test$price, prediction = pred4$net.result)
results.test2
RMSE(pred4$net.result, test$price)


## -----------------------------------------------------------------------------------------------------------
unscaled3 <- (pred3$net.result) * sd(train1$price) + mean(train1$price)
unscaled3
results.train3 <- data.frame(actual = train1$price, prediction = unscaled3)
results.train3


## -----------------------------------------------------------------------------------------------------------
unscaled4 <- (pred4$net.result) * sd(test1$price) + mean(test1$price)
unscaled4
results.test3 <- data.frame(actual = test1$price, prediction = unscaled4)
results.test3


## -----------------------------------------------------------------------------------------------------------
plot(results.train3, col='blue', pch=16, main = "predicted vs actual for the training set", ylab = "predicted", xlab = "actual")
plot(results.test3, col='blue', pch=16, main = "predicted vs actual for the testing set", ylab = "predicted", xlab = "actual")


## -----------------------------------------------------------------------------------------------------------
folds <- createFolds(train$price, k = 10)
str(folds)
results <- c()
for (fld in folds){
  index <- sample(1:nrow(train),round(0.9*nrow(train)))
  data <- train[-fld,]
  nn2 <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody +
    drivewheel + enginelocation + wheelbase + carlength + carwidth +
    carheight + curbweight + enginetype + cylindernumber + enginesize +
    fuelsystem + boreratio + stroke + compressionratio + horsepower +
    peakrpm + citympg + highwaympg, data = train, hidden = c(6,4))
  pred.val3 <- compute(nn2, train[,1:23])
  results <- cbind(results,RMSE(pred.val3$net.result, train$price))
}
paste("After", length(results), "validation loops the mean error of the network is", paste0(round(mean(results),2), "%"))


## -----------------------------------------------------------------------------------------------------------
folds <- createFolds(test$price, k = 10)
str(folds)
results <- c()
for (fld in folds){
  index <- sample(1:nrow(test),round(0.9*nrow(test)))
  data <- test[-fld,]
  nn2 <- neuralnet(price ~ symboling + fueltype + aspiration + doornumber + carbody +
    drivewheel + enginelocation + wheelbase + carlength + carwidth +
    carheight + curbweight + enginetype + cylindernumber + enginesize +
    fuelsystem + boreratio + stroke + compressionratio + horsepower +
    peakrpm + citympg + highwaympg, data = test, hidden = c(6,4))
  pred.val4 <- compute(nn2, test[,1:23])
  results <- cbind(results,RMSE(pred.val4$net.result, test$price))
}
paste("After", length(results), "validation loops the mean error of the network is", paste0(round(mean(results),2), "%"))

