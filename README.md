# ML_Project

---
title: "COVID19 Machine learning Project"
author: 'Nurarlisa Sulong : 17220304'
output: html_document
---
### Introduction
#### Assignment Title
Using machine learning approaches (Logistic regression, Decision tree) to predictive modelling the spread of the COVID-19 pandemic 

#### Objective
1. To generate a model of coronavirus disease spread using machine learning methods.
2. To compare logistic regression and a decision tree to predict whether the number of cases in a country can be used to predict recovery of a patient infected by coronavirus

#### Data source
The public available dataset “2019 Novel Coronavirus Data Repository” published by Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE)

The GITHUB link to this Project is [here](https://github.com/CSSEGISandData/COVID-19)

The objectives want to investigate the following:

Is the chance of recovery lower/higher for someone infected by corona virus based on the number of cases in a country or where they were infected ?

A target variable “recovery” was computed with 1 indicating a recovered person and 0 indicating someone who is confirmed as infected or died from coronavirus.

#### Data Accessibility
The coronavirus package provides a tidy format dataset of the 2019 Novel Coronavirus COVID-19 (2019-nCoV) epidemic. The packaged was released by Rami Krispin on https://github.com/RamiKrispin/coronavirus

### Installation
```r
install.packages("rmarkdown")
library(rmarkdown)

install.packages("knitr")
library(knitr)

#install.packages("devtools")
devtools::install_github("RamiKrispin/coronavirus", force = TRUE)

install.packages("tidyverse")
install.packages("magrittr")
library(magrittr)
install.packages("ROCR")
```

```{r, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE, 
  comment = "#>",
  fig.path = "man/figures/README-",
  out.width = "100%",
  message=FALSE, 
  warning=FALSE
)
library(coronavirus)
``` 

### Usage
```{r}
data("coronavirus")
```

**Explore the data by look at the column names and check the first 6 rows of data**
```{r}
names(coronavirus)
head(coronavirus)

```
**Creat data frame by filling date from 22 November 2020 until 26 December 2020 (5 weeks)**
```{r}
install.packages("dplyr",repos = "http://cran.us.r-project.org")
library(dplyr)

Coronavirus <- coronavirus %>% filter(date > "2020-11-22" & date < "2020-12-26")
```

**Summary of the total confrimed cases by country (top 20)**
```{r}
summary_df <- Coronavirus %>% 
  filter(type == "confirmed") %>%
  group_by(country) %>%
  summarise(total_cases = sum(cases)) %>%
  arrange(-total_cases)

summary_df %>% head(20) 
```
**Create New Column recovery**
```{r}
recovery <- c(1:26334)
cbind(Coronavirus,recovery)
Coronavirus$recovery[Coronavirus$type == "recovered"]<-1
Coronavirus$recovery[Coronavirus$type != "recovered" ]<-0
```
```{r}
names(Coronavirus)
```

**Tabulation of recovery**
```{r}
table(Coronavirus$recovery)
```

**Splitting Training & Testing Data**

Split-out validation dataset.Given that there is only one data set, it is paramount that we randomly split the data set into a training set and testing set


  Split the data with a split ratio, 75% of the data in the training set (use to build the model) and 25% of the data in the testing
```{r}
library(caTools)
```

```{r}
set.seed(1)
split = sample.split(Coronavirus$recovery, SplitRatio = 0.85)

dfTrain = subset(Coronavirus, split == TRUE)
dfTest = subset(Coronavirus, split == FALSE)
```

**Tabulate recovery variable outcomes**
```{r}
table(dfTrain$recovery)
table(dfTest$recovery)
```


```{r}
dfTrain$recovery<-factor(dfTrain$recovery)
```

**Feature Selection**
```{r}
install.packages('caret', repos='http://cran.rstudio.com/')
library(caret)
```

```{r}
set.seed(7)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
```
**train the model**
```{r}
model <- train(recovery~cases + country, data=dfTrain, method="glm", preProcess="scale", trControl=control)
```
**estimate variable importance and summarize**
```{r}
importance <- varImp(model, scale=FALSE)
print(importance)
```

### Logistic regression assumptions
**1. Linearity assumption**
```{r}
logistic = glm(recovery ~ cases+country,data=dfTrain, family="binomial")
summary(logistic)
```

**2. Multicollinearity**
```{r}
car::vif(logistic)
```

**3.Check for Influential Outliers**
```{r}
library(tidyverse)
library(broom)
```

```{r}
plot(logistic, which = 4, id.n = 3)
```

**Extract model results**
```{r}
logistic.data <- augment(logistic) %>% 
  mutate(index = 1:n()) 

logistic.data %>% top_n(3, .cooksd)
```

```{r}
ggplot(logistic.data, aes(index, .std.resid)) + 
  geom_point(aes(color = recovery), alpha = .5) +
  theme_bw()
```

  **Making predictions on Training set**
```{r}
predicted = data.frame(probability.of.recovery= logistic$fitted.value, recovery=dfTrain$recovery)
predicted = predicted[order(predicted$probability.of.recovery, decreasing = FALSE),]
predicted$rank = 1:nrow(predicted)
library(ggplot2)
library(cowplot)
```

```{r}
ggplot(data=predicted, aes(x=rank, y=probability.of.recovery)) + 
  geom_point(aes(colour=recovery), alpha=1, shape=4, stroke=2)+
  xlab("Index") +
  ylab("Predicted Probability of Recovery")
```

**Confusion matrix**
```{r}
predictTrain = predict(logistic, type="response")
tapply(predictTrain, dfTrain$recovery, mean)
table(dfTrain$recovery, predictTrain > 0.4)
```

**Classification Accuracy**
```
a = 15203
b = 0
c = 7180
d = 1

(a+d)/(a+b+c+d)
```

===========================================================================

### Decision Trees
```
install.packages('rattle')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('RColorBrewer')

library(rattle, warn.conflicts = FALSE)
library(rpart.plot, warn.conflicts = FALSE)
library(RColorBrewer, warn.conflicts = FALSE)

```

**Changing the prior probabilities of recovery **
```{r}
tree_prior = rpart(recovery ~ cases + country, method = "class",
                    data = dfTrain,
                    parms = list(prior = c(0.7,0.3)),
                    control = rpart.control(cp = 0.001))
prp(tree_prior)
```

**cp and xerror values and plot**
```{r}
 printcp(tree_prior)
 plotcp(tree_prior) 
```

**pruning the tree for increased model performance**
```{r}
 tree_min = tree_prior$cptable[which.min(tree_prior$cptable[,"xerror"]),"CP"]
ptree_prior = prune(tree_prior, cp = tree_min) # pruning the tree.
prp(ptree_prior)
```

**making predictions**
```{r}
pred_prior = predict(ptree_prior, newdata = dfTest, type = "class")
```

**Confusion matrix**
```{r}
confmat_prior = table(dfTest$recovery,pred_prior)
confmat_prior
```
```{r}
acc_prior = sum(diag(confmat_prior)) / sum(confmat_prior)
acc_prior
```

**Case weights for the training dataset**
```{r}
case_weights = ifelse(dfTrain$recovery == 0,1,3)

tree_weights = rpart(recovery ~ cases + country, method = "class", 
                      data = dfTrain, 
                      control = rpart.control(cp = 0.001,minsplit = 5,minbucket = 2),
                      weights = case_weights)

plotcp(tree_weights)
```

```{r}
ptree_weights = prune(tree_weights, cp = 0.00183101)
prp(ptree_weights,extra = 1)
```

```{r}
pred_weights = predict(ptree_weights, newdata = dfTest,type = "class")
confmat_weights = table(dfTest$recovery,pred_weights)
confmat_weights
```

```{r}
acc_weights = sum(diag(confmat_weights)) / sum(confmat_weights)
acc_weights
```


## Conclusion

It can be seen that most of the cases in the data did not recover. A logistic regression and a decision tree algorithm were each applied to the data in an effort to optimally predict whether a patient would recover based on the frequency of cases in their country.

From the results in the analysis it can be seen that the logistic regression method produced better results than Decision trees with respect to Accuracy at 68% success rate at prediction.

