library(tidyverse)
library(vip)
library(xgboost)
library(rsample)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)

set.seed(47)

titanic_df <- titanic |>
  mutate(pclass = as.factor(pclass)) |>
  drop_na() 

# randomly subsetting positive class rows to have larger class imbalanced
to_remove <- sample(which(titanic_df$survived == 'yes'),
                    floor(length(which(titanic_df$survived == 'yes')) * 0.8))

titanic_sample <- titanic_df[-to_remove,]

table(titanic_sample$survived)

# randomly splitting subset 

titanic_split <- initial_split(titanic_sample)
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

# transforming data so that it works with xgboost/rpart

X_train <- model.matrix(survived ~. -1, data = titanic_train)
y_train <- as.numeric(titanic_train$survived) - 1

X_test <- model.matrix(survived ~. -1, data = titanic_test)
y_test <- as.numeric(titanic_test$survived) - 1

xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

# CV grid/settings
xgboost_tune_grid <- expand.grid(nrounds = seq(from = 20, to = 200, by = 20),
                                 eta = c(0.025, 0.05, 0.1, 0.3),
                                 gamma = 0,
                                 max_depth = c(1, 2, 3, 4),
                                 colsample_bytree = 1,
                                 min_child_weight = 1,
                                 subsample = 1)

xgboost_tune_control <- trainControl(method = "cv",
                                     number = 5,
                                     classProbs = TRUE,
                                     summaryFunction=mnLogLoss)

# CV on decision tree and xgboost
dt_tune <- caret::train(x = titanic_train[,-1], y = titanic_train$survived,
                     method = "rpart",
                     metric = "logLoss",
                     trControl = xgboost_tune_control,
                     tuneLength = 20)

xgboost_tune <- caret::train(x = X_train, y = titanic_train$survived, 
             method = "xgbTree",
             metric = "logLoss",
             trControl = xgboost_tune_control,
             tuneGrid = xgboost_tune_grid,
             verbose = 0)

# training xgboost with CV parameters

xgboost_fit_final <- xgboost(data = xgboost_train,
        objective = "binary:logistic",
        eval_metric = c("error", "auc"),
        nrounds = xgboost_tune$bestTune$nrounds,
        params = as.list(dplyr::select(xgboost_tune$bestTune, -nrounds)),
        verbose = 0)

# selecting xgboost threshold using training data

xgbprob_train <- predict(xgboost_fit_final, newdata = xgboost_train, type = 'prob')
xgbroc_train <- roc(titanic_train$survived, xgbprob_train)
xgb_coords <- coords(xgbroc_train, "best", best.method = "closest.topleft") 
xgb_best_threshold <- xgb_coords$threshold
xgb_best_threshold

xgbprob_test <- predict(xgboost_fit_final, newdata = xgboost_test, type = 'prob')
xgbpred_test <- ifelse (xgbprob_test > xgb_best_threshold, 'yes', 'no')

# confusion matrix comparison on test data

xgboost_cm_df <- data.frame(prediction = as.factor(xgbpred_test), 
                 actual = as.factor(titanic_test$survived))
dtpred_test <- predict(dt_tune, titanic_test)


confusionMatrix(dtpred_test, titanic_test$survived, positive = "yes")
confusionMatrix(xgboost_cm_df$prediction, xgboost_cm_df$actual, positive = "yes")

dtprob_test <- predict(dt_tune, titanic_test, type = "prob")

# decision tree visualization

rpart.plot(dt_tune$finalModel)

# test ROC curve comparison
dtroc_test <- roc(titanic_test$survived, dtprob_test$yes)
xgbroc_test <- roc(titanic_test$survived, xgbprob_test)

roc_data <- data.frame(model = c(rep('Decision Tree', length(dtroc_test$sensitivities)),
                     rep('XGBoost', length(xgbroc_test$sensitivities))),
           sensitivity = c(dtroc_test$sensitivities,
                           xgbroc_test$sensitivities),
           specificity = c(dtroc_test$specificities,
                           xgbroc_test$specificities))

roc_data |>
  ggplot(aes(x = 1- specificity, y = sensitivity))+
  geom_line(linewidth = 0.7, aes(color = model))+
  geom_abline(intercept = 0, slope = 1)+
  labs(x = 'False Positive Rate', y = 'True Positive Rate')+
  scale_color_brewer(palette = 'Set1')+
  guides(color = guide_legend('Model Type'))+
  theme_minimal()+
  theme(strip.text = element_text(color = 'black', face = 'bold'),
        axis.title = element_text(color = 'black'),
        axis.text = element_text(color = 'black'))

# saving ROC curve figure to working directory
ggsave('roc-complete.png', width = 6, height = 4, bg = 'white')

