Practical Machine Learning -- Course Project
============================================

### Introduction
The goal of the project is to predict the manner of the exercise execution from the personal activity data collected using accelerometers on the belt, forearm, arm and dumbell of 6 participants.

The data files [named **pml-training.csv** and **pml-testing.csv**] were provided in the assignment desciption. Originally they come from http://groupware.les.inf.puc-rio.br/har.

The training data file contains 160 variables. THe outcome variable which denotes the exercise execution classificaion is *classe*.

### Model building
The model building is explained with provided R code and some remarks for its meaning.

- Begin by loading neccesary libraries

```r
library(caret)
library(randomForest)
```

- Read the data from the files which are assumed to be in a current directory. Note the string "#DIV/0!" in the **na.strings** argument which during exploratory analysis was recognized to be present in the data and meaning nonpresent value

```r
dataTraining <- read.csv("pml-training.csv",header=TRUE,sep=",",na.strings=c("NA","#DIV/0!"))
dataTesting <- read.csv("pml-testing.csv",header=TRUE,sep=",",na.strings=c("NA","#DIV/0!"))
```

- Split the data into training and test set. The latter will be used to predict the out of sample error.

```r
inTrain <- createDataPartition(y=dataTraining$classe,p=0.75,list=FALSE)
training <- dataTraining[inTrain,]
testing <- dataTraining[-inTrain,]
```

- During initial exploration of the data, it was noticed the existance of large number of NA values in various variables. It was considered to use some imputing method, but for simplicity the chosen way was to simply remove them. In addition, first six variables are removed from the predictors because they are dubious. Also, initially all logical and factor variables are subtracted -- they will not be used as predictors.

```r
names <- names(dataTraining)
r <- vector("integer") #vector holding indices of variables that are removed from used predictors
r <- 1:6
k <- 7
for(i in k:dim(dataTraining)[2]) {
  if(sum(is.na(training[,i]))>0 | (class(training[,i]) %in% c("logical","factor"))) {
    r[k] <- i
    k <- k+1
  }
}
training <- training[,-r]
testing <- testing[,-r]
testing2 <- dataTesting[,-r]
```

- Check the class of predictor variables.

```r
names <- names(training)
c <- vector("character",length(names))
for(i in seq_along(names)) {
  c[i] <- class(training[[names[i]]])
}
table(c)
```

```
## c
## integer numeric 
##      26      27
```

- Preprocess predictors by normalizing them.

```r
preProc <- preProcess(training,method=c("center","scale"))
training <- predict(preProc,training)
testing <- predict(preProc,testing)
testing2 <- predict(preProc,testing2)
```

- Restore the outcome variable *classe* into the data sets

```r
training[,"classe"] <- dataTraining[inTrain,"classe"]
testing[,"classe"] <- dataTraining[-inTrain,"classe"]
```

- The algorithm for classification was selected to be **Random Forests** because of its high accuracy. For model building the **caret** package function *train()* was not used because of its slowness. Function *randomForest* was used instead.

```r
fit <- randomForest(classe~.,data=training,ntree=500)
```

- Test the model for the training data.

```r
p <- predict(fit,training)
confusionMatrix(p,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

- Get the predictive results of the model generally.

```r
p2 <- predict(fit,testing)
confusionMatrix(p2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  946    1    0    0
##          C    0    1  853    1    0
##          D    0    0    1  801    0
##          E    0    0    0    2  901
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.998         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.997    0.998    0.996    1.000
## Specificity             0.999    1.000    1.000    1.000    1.000
## Pos Pred Value          0.999    0.999    0.998    0.999    0.998
## Neg Pred Value          1.000    0.999    1.000    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.998    0.999    0.998    1.000
```

- Get the preditive results for the 20 observations of the test data.

```r
predict(fit,testing2)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


### Conclusion
Despite some simplifying decisions the model achieves very high predictive accuracy of 0.9984.
