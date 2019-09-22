rm(list = ls())
library(readr)
library(xts)
library(forecast)

# Datapath

setwd('/Users/mrinmoysarkar/Downloads/Hourly Data')

#-----------------------------------
# Load Complete Data

load_2009 <- read_csv("load 2009.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2009)

load_2010 <- read_csv("load 2010.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2010)

load_2011 <- read_csv("load 2011.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2011)

load_2012 <- read_csv("load 2012.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2012)

load_2013 <- read_csv("load 2013.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2013)

load_2014 <- read_csv("load 2014.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2014)

load_2015 <- read_csv("load 2015.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2015)

load_2016 <- read_csv("load 2016.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2016) # Contains Missing Data

load_2017 <- read_csv("load 2017.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2017)

load_2018 <- read_csv("load 2018.csv", 
                      col_types = cols(Hour_End = col_datetime(format = "%m/%d/%Y %H:%M")))
summary(load_2018)

#---------------------------------
# Load Testbed
y.2009 <- xts((load_2009$WEST),order.by = load_2009$Hour_End - 3599)
y.2010 <- xts((load_2010$WEST),order.by = load_2010$Hour_End - 3599)
y.2011 <- xts((load_2011$WEST),order.by = load_2011$Hour_End - 3599)
y.2012 <- xts((load_2012$WEST),order.by = load_2012$Hour_End - 3599)
y.2013 <- xts((load_2013$WEST),order.by = load_2013$Hour_End - 3599)
y.2014 <- xts((load_2014$WEST),order.by = load_2014$Hour_End - 3599)
y.2015 <- xts((load_2015$WEST),order.by = load_2015$Hour_End - 3599)
y.2016 <- xts((load_2016$WEST),order.by = load_2016$Hour_End - 3599)
y.2017 <- xts((load_2017$WEST),order.by = load_2017$Hour_End - 3599)
y.2018 <- xts((load_2018$WEST),order.by = load_2018$Hour_End - 3599)

#----------------------------------------
# Daily Load Data transformation

daily.2009 <- apply.daily(y.2009, median)
daily.2010 <- apply.daily(y.2010, median)
daily.2011 <- apply.daily(y.2011, median)
daily.2012 <- apply.daily(y.2012, median)
daily.2013 <- apply.daily(y.2013, median)
daily.2014 <- apply.daily(y.2014, median)
daily.2015 <- apply.daily(y.2015, median)
daily.2016 <- apply.daily(y.2016, median)
daily.2017 <- apply.daily(y.2017, median)
daily.2018 <- apply.daily(y.2018, median)

#----------------------------------------
# Generate .csv files for daily data

write.csv(data.frame(Time = index(daily.2009), Power = as.numeric(daily.2009)), file = 'daily_2009_west.csv')
write.csv(data.frame(Time = index(daily.2010), Power = as.numeric(daily.2010)), file = 'daily_2010_west.csv')
write.csv(data.frame(Time = index(daily.2011), Power = as.numeric(daily.2011)), file = 'daily_2011_west.csv')
write.csv(data.frame(Time = index(daily.2012), Power = as.numeric(daily.2012)), file = 'daily_2012_west.csv')
write.csv(data.frame(Time = index(daily.2013), Power = as.numeric(daily.2013)), file = 'daily_2013_west.csv')
write.csv(data.frame(Time = index(daily.2014), Power = as.numeric(daily.2014)), file = 'daily_2014_west.csv')
write.csv(data.frame(Time = index(daily.2015), Power = as.numeric(daily.2015)), file = 'daily_2015_west.csv')
write.csv(data.frame(Time = index(daily.2016), Power = as.numeric(daily.2016)), file = 'daily_2016_west.csv')
write.csv(data.frame(Time = index(daily.2017), Power = as.numeric(daily.2017)), file = 'daily_2017_west.csv')
write.csv(data.frame(Time = index(daily.2018), Power = as.numeric(daily.2018)), file = 'daily_2018_west.csv')

#------------------------------------
# Missing Data Treatment using interpolation

summary(daily.2016)
daily.2016 <- na.approx(daily.2016)

#--------------------------------------------------------
# Merge 9-Years Load Data to Train and Evaluate Prediction Errors of 2018

train.Y <- rbind( daily.2009, daily.2010, daily.2011, daily.2012, daily.2013, daily.2014, 
                  daily.2015, daily.2016, daily.2017)

#-----------------------------------
# Merge 9-Years Load Data and 2018 Data upto November 30 to Train and Predict December, 2018 and Entire 2019

#train.Y <- rbind( daily.2009, daily.2010, daily.2011, daily.2012, daily.2013, daily.2014, 
#                  daily.2015, daily.2016, daily.2017, daily.2018)

x <- ts(train.Y, start = c(2009,1), frequency = 365)
ma.x <- ma(train.Y, 7)
ma.x <- ts(ma.x, start = c(2009,1), frequency = 365)

summary(train.Y)

# Forecast period
fp <- seq(as.Date('2018-12-01 23:00:01 UTC'), as.Date('2019-12-31 23:00:01 UTC'), length.out = 334) 

#-----------------------------------------------------------------
# Multi seasonality object and plots-----------------------

m.series <- msts(train.Y, seasonal.periods=c(365.25), start = c(2009,1))

# Decomposition plot
plot(m.series)
autoplot(m.series)
plot(decompose(m.series))
autoplot(ma.x, main = 'Moving average Power', ylab = 'Power (log(Kw))')
plot(train.Y)
autoplot(train.Y)
plot(decompose(x))
autoplot(decompose(x))

#Multiple seasonality adjusted
plot(mstl(x))
autoplot(mstl(x))

plot(x)
autoplot(x)
plot(decompose(x))

m.series <- msts(train.Y, seasonal.periods=c(30, 365.25), start = c(2009,1))

plot(tbats(m.series))

# Algorithms to Test

#----------------------------------------------
# Auto-Regressive Integrated Moving Average (ARIMA) Algorithm

m.series <- msts(train.Y, seasonal.periods=c(365.25), start = c(2009,1))
arima.fit <- auto.arima(m.series, allowdrift= T, allowmean = T)
# h = 334 when calculating test error
# h = 396 when forecasting Dec 2018 and 2019
pred <- forecast(arima.fit, h = 334)
plot(pred)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred$mean), file = 'Prediction_ARIMA_west.csv') 
write.csv(data.frame(Power = pred$x), file = 'Train_ARIMA_west.csv') 

plot(xts(pred$mean, order.by = index(daily.2018)))
accuracy(pred,daily.2018)
ts.plot(daily.2018,ts(pred$mean, start = 0), col=2:3)
plot(daily.2018, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(pred$mean, order.by = index(daily.2018)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)
checkresiduals(arima.fit)

#--------------------------------------------
# Exponential Smoothing State Space Algorithm

m.series <- msts(train.Y, seasonal.periods=c(7, 30,365.25), start = c(2009,1))
# Test for multiple seasoning
# m.series <- msts(train.Y, seasonal.periods=c(7, 365.25), start = c(2009,1))
# m.series <- msts(train.Y, seasonal.periods=c(30, 365.25), start = c(2009,1))
tbats.fit <- tbats(m.series, use.box.cox = T, use.trend = T)
# h = 334 when calculating test error
# h = 396 when forecasting Dec 2018 and 2019
pred <- forecast(tbats.fit, h = 334)
plot(pred)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred$mean), file = 'Prediction_ExpSmooth_west.csv')

accuracy(pred,daily.2018)
ts.plot(daily.2018,ts(pred$mean, start = 0), col=2:3)
plot(daily.2018, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(pred$mean, order.by = index(daily.2018)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)
checkresiduals(tbats.fit)

#--------------------------------------
# Linear Regression Algorithm

m.series <- msts(train.Y, seasonal.periods=c(365.25), start = c(2009,1))
tslm.fit <- tslm(m.series ~ trend + season)
# h = 334 when calculating test error
# h = 396 when forecasting Dec 2018 and 2019
pred <- forecast(tslm.fit, h = 334)
plot(pred)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred$mean), file = 'Prediction_Linear_west.csv')

accuracy(pred,daily.2018)
#accuracy(exp(pred$mean),exp(as.numeric(daily.2018)))
ts.plot(daily.2018,ts(pred$mean, start = 0), col=2:3)
plot(daily.2018, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(pred$mean, order.by = index(daily.2018)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)
checkresiduals(tslm.fit)

#Ploting forecast
plot(pred, include = 10)

#----------------------------------------
# Basic Neural Network Algorithm

nnet.fit <- nnetar(m.series)
# h = 334 when calculating test error
# h = 396 when forecasting Dec 2018 and 2019
pred <- forecast(nnet.fit, h = 334)
plot(pred)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred$mean), file = 'Prediction_NN_west.csv')

accuracy(pred,daily.2018)

ts.plot(daily.2018,ts(pred$mean, start = 0), col=2:3)

#-------------------------------------------
# Random Forest Algorithm

day <- as.numeric(format(index(train.Y), "%d"))
mon <- as.numeric(format(index(train.Y), "%m"))
year <- as.numeric(format(index(train.Y), "%y"))

X <- cbind(day, mon, year)

y <- as.numeric(train.Y)

# For calculating test error---------------------

t.day <- as.numeric(format(index(daily.2018), "%d"))
t.mon <- as.numeric(format(index(daily.2018), "%m"))
t.year <- as.numeric(format(index(daily.2018), "%y"))
t.X <- cbind(t.day, t.mon, t.year)
t.y <- as.numeric(daily.2018)
colnames(t.X) <- colnames(X)

# For forecasting -------------------

f.day <- c(1:31, as.numeric(format(index(daily.2017), "%d")))
f.mon <- c(rep(12,31), as.numeric(format(index(daily.2017), "%m")))
f.year <- c(rep(18, 31), rep(19, 365))
f.X <- cbind(f.day, f.mon, f.year)
colnames(f.X) <- colnames(X)

#----------------------------

library(randomForest)

rf <- randomForest(x = X, y = y, xtest = t.X, yest = t.y, ntree = 1024, keep.forest=TRUE)

fitted <- rf$predicted
fitted.ts <- ts(fitted, start = c(2009,1), frequency = 365)

pred <- predict(rf, t.X)
rf.ts <- ts(pred, start = c(2018,1), frequency = 365)

accuracy(y, fitted.ts) # training error
accuracy(t.y, rf.ts) # testing error

plot(train.Y, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(fitted, order.by = index(train.Y)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)

plot(daily.2018, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(rf.ts, order.by = index(daily.2018)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)

# Forecast with Random Forest

pred <- predict(rf, f.X)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred), file = 'Prediction_RF_west.csv')

rf.ts <- ts(pred, start = c(2018,12), frequency = 365)
plot(ts(rf.ts, start = c(2018,12), frequency = 365), lwd = 2, col= 3, main='Time series plot')

#---------------------
# Support Vector Machine (SVM) Algorithm

library(kernlab)

model <- ksvm(y ~ X, kernel ='rbfdot')

pred <- predict(model, newdata = t.X)
pred <- ts(pred, start = c(2018,1), frequency = 365)

plot(pred)

fitted <- predict(model, newdata = X)
fitted.ts <- ts(fitted, start = c(2009,1), frequency = 365)

plot(train.Y, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(fitted, order.by = index(train.Y)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)

plot(daily.2018, lwd = 2, col= 3, main='Time series plot')
addSeries(xts(pred, order.by = index(daily.2018)), on = 1, col = 2, lwd = 2)
addLegend('bottomright', c('True', 'Predicted'), lty = 1 , col=3:2)


accuracy(y, fitted) # training error
accuracy(t.y, pred) # testing error

# Forecast with SVM

pred <- predict(model, newdata = f.X)

# Make .csv file for the forecast
write.csv(data.frame(Time = fp, Power = pred), file = 'Prediction_SVM_west.csv')

pred <- ts(pred, start = c(2018,12), frequency = 365)

plot(ts(pred, start = c(2018,12), frequency = 365), lwd=2, col=3, main='Time series plot')





