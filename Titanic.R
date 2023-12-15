library(ggplot2)
library(scales)

train$SibSp <- as.factor(train$SibSp)
train$Parch <- as.factor(train$Parch)
train$Pclass <- as.factor(train$Pclass)
train$Sex <- as.factor(train$Sex)
train$Survived <- as.factor(train$Survived)

test$SibSp <- as.factor(test$SibSp)
test$Parch <- as.factor(test$Parch)
test$Pclass <- as.factor(test$Pclass)
test$Sex <- as.factor(test$Sex)

ggplot(train, aes(x = Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex) +
  scale_y_continuous(breaks = seq(0,1,0.1)) +
  labs(y = "Percentage of Passangers", title = "Survival proportion per class and sex")

ggplot(train, aes(x = SibSp, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex + Pclass) +
  labs(y = "Percentage of Passangers", title = "Survival per SibSp proportion, Pclass and sex")

ggplot(train, aes(x = Parch, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex + Pclass) +
  labs(y = "Percentage of Passangers", title = "Survival per Parch proportion, Pclass and sex")

ggplot(train, aes(x = Embarked, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex + Pclass) +
  scale_y_continuous(breaks = seq(0,1,0.1)) +
  labs(y = "Percentage of Passangers", title = "Survival per Emabarked proportion, Pclass and sex")

# Male survival is 1st = 37,5%, 2nd = 16%, 3rd = 13%
# Female survival is 1st = 96%, 2nd = 92%, 3rd = 50%
# -> Embarked doesn't seem to play a role in survival

train.cabin <- train[complete.cases(train$Cabin),]

# Cabin might be interesting due to the fact that the more below deck cabin might mean the a lower chance to get
# on deck. However, there are WAY too many missing values and there seems to be a large spread of cabins in 1st class,
# everything from A-E, making it impossible to substitute NA values with something plausible.
# A possible substitute for cabin might be the fare price...

test.survived <- data.frame(Survived = rep("None", nrow(test)), test[,])
train.complete <- rbind(train, test.survived)

ticket.stats <- train.complete %>%
  group_by(Ticket) %>%
  summarize(Group.Count = n(),
            Avg.Fare = max(Fare) / n())

train.complete <- train.complete %>%
  left_join(ticket.stats, by = "Ticket")

ggplot(train.complete[1:891,], aes(x = Avg.Fare, fill = Survived)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~Sex) +
  scale_x_continuous(breaks = seq(0,250,5), minor_breaks = seq(0, 250, 1)) +
  labs(y = "Number of Passangers", title = "Survival per average ticket price")

train.plot <- train.complete[1:891,] %>%
  mutate(fare.bin = cut(Avg.Fare, breaks = seq(0, length(Avg.Fare), by = 1), labels = FALSE, include.lowest = TRUE)) %>%
  count(Survived, fare.bin) %>%
  group_by(fare.bin) %>%
  mutate(percent = n/sum(n))

ggplot(train.plot, aes(x = fare.bin, y = percent, fill = Survived)) +
  geom_col(position = "stack") +
  scale_x_continuous(breaks = seq(0,70,2))

# Survival 100% when ticket price is 60£ or more
# Survival goes from 25% at ticket price 0-6£ to 62,5% at ticket price 6-60£

library(rpart)
library(rpart.plot)
library(caret)
library(doSNOW)
library(e1071)

rpart.cv <- function(seed, training, labels, ctrl) {
  cl <- makeCluster(4, type = "SOCK")
  registerDoSNOW(cl)

  set.seed(seed)
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30, trControl = ctrl)

  stopCluster(cl)

  return (rpart.cv)
}

rf.label <- as.factor(train$Survived)
cv.10.folds <- createMultiFolds(rf.label, k = 10, times = 10)
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, index = cv.10.folds)

# "Pclass", "Sex", "Avg.Fare" = 0.8067165
features <- c("Pclass", "Sex", "Avg.Fare")
rpart.train.1 <- train.complete[1:891, features]

rpart.1.cv.1 <- rpart.cv(1234, rpart.train.1, rf.label, ctrl.1)
rpart.1.cv.1

prp(rpart.1.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex", "SibSp" = 0.7937089
features <- c("Pclass", "Sex", "SibSp")
rpart.train.2 <- train.complete[1:891, features]

rpart.2.cv.1 <- rpart.cv(1234, rpart.train.2, rf.label, ctrl.1)
rpart.2.cv.1

prp(rpart.2.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex", "Parch" = 0.7924639
features <- c("Pclass", "Sex", "Parch")
rpart.train.3 <- train.complete[1:891, features]

rpart.3.cv.1 <- rpart.cv(1234, rpart.train.3, rf.label, ctrl.1)
rpart.3.cv.1

prp(rpart.3.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex", "SibSp", "Parch", "Avg.Fare" = 0.8016979
features <- c("Pclass", "Sex", "SibSp", "Parch", "Avg.Fare")
rpart.train.4 <- train.complete[1:891, features]

rpart.4.cv.1 <- rpart.cv(1234, rpart.train.4, rf.label, ctrl.1)
rpart.4.cv.1

prp(rpart.4.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex" = 0.7867398
features <- c("Pclass", "Sex")
rpart.train.5 <- train.complete[1:891, features]

rpart.5.cv.1 <- rpart.cv(1234, rpart.train.5, rf.label, ctrl.1)
rpart.5.cv.1

prp(rpart.5.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex", "Group.Count" = 0.8114384
features <- c("Pclass", "Sex", "Group.Count")
rpart.train.6 <- train.complete[1:891, features]

rpart.6.cv.1 <- rpart.cv(1234, rpart.train.6, rf.label, ctrl.1)
rpart.6.cv.1

prp(rpart.6.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# "Pclass", "Sex", "Group.Count", "Avg.Fare" = 0.8189604
features <- c("Pclass", "Sex", "Group.Count", "Avg.Fare")
rpart.train.7 <- train.complete[1:891, features]

rpart.7.cv.1 <- rpart.cv(1234, rpart.train.7, rf.label, ctrl.1)
rpart.7.cv.1

prp(rpart.7.cv.1$finalModel, type = 0, extra = 1, under = TRUE)

# RandomForest training
library(randomForest)

rf.label <- as.factor(train$Survived)

# "Pclass", "Sex" =     0   1 class.error
#                       0 485  64   0.1165756
#                       1 169 173   0.4941520
rf.train.1 <- train.complete[1:891, c("Pclass", "Sex")]
set.seed(1234)

rf.1 <- randomForest(x = rf.train.1, y = rf.label, importance = TRUE, ntree = 1000)
rf.1
varImpPlot(rf.1)


rf.train.2 <- train.complete[1:891, c("Pclass", "Sex", "Group.Count", "Avg.Fare")]
set.seed(1234)

rf.2 <- randomForest(x = rf.train.2, y = rf.label, importance = TRUE, ntree = 1000)
rf.2
varImpPlot(rf.2)

# Replace the missing fare value by the average fare for that Pclass and Sex
library(plyr)
sum(is.na(train.complete$Avg.Fare))
ddply(train.complete, .(Pclass, Sex), summarize, Avg.Fare.mean = mean(Avg.Fare, na.rm = TRUE))

index <- which(is.na(train.complete$Avg.Fare))
train.complete$Avg.Fare[index] <- 7.449765

# Submit data
test.submit.df <- train.complete[892:1309, c("Pclass", "Sex", "Group.Count", "Avg.Fare")]
rf.2.preds <- predict(rf.2, test.submit.df)
table(rf.2.preds)

submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rf.2.preds)
write.csv(submit.df, file = "RF_SUB_20170408_2.csv", row.names = FALSE)

# Cross-validation 10-fold
set.seed(2345)
cv.10.folds = createMultiFolds(rf.label, k = 10, times = 10)

ctrl.2 <- trainControl(method = "repeatedcv", number = 10, repeats = 10, index = cv.10.folds)

cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

set.seed(3456)
rf.2.cv.2 <- train(x = rf.train.2, y = rf.label, method = "rf", tuneLength = 3, ntree = 1000, trControl = ctrl.2)

stopCluster(cl)

rf.2.cv.2

# Cross-validation 5-fold
set.seed(4567)
cv.5.folds <- createMultiFolds(rf.label, k = 5, times = 10)

ctrl.3 <- trainControl(method = "repeatedcv", number = 5, repeats = 10, index = cv.5.folds)

cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

set.seed(5678)
rf.3.cv.3 <- train(x = rf.train.2, y = rf.label, method = "rf", tuneLength = 3, ntree = 1000, trControl = ctrl.3)

stopCluster(cl)

rf.3.cv.3

# Cross-validation 3-fold
set.seed(6789)
cv.3.folds <- createMultiFolds(rf.label, k = 3, times = 10)

ctrl.4 <- trainControl(method = "repeatedcv", number = 3, repeats = 10, index = cv.3.folds)

cl <- makeCluster(4, type = "SOCK")
registerDoSNOW(cl)

set.seed(7890)
rf.4.cv.4 <- train(x = rf.train.2, y = rf.label, method = "rf", tuneLength = 3, ntree = 1000, trControl = ctrl.4)

stopCluster(cl)

rf.4.cv.4

# Try to improve the model
rpart.cv <- function(seed, training, labels, ctrl) {
  cl <- makeCluster(4, type = "SOCK")
  registerDoSNOW(cl)

  set.seed(seed)
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30, trControl = ctrl)

  stopCluster(cl)

  return (rpart.cv)
}

# 3-fold "Pclass", "Sex", "Avg.Fare" =
features <- c("Pclass", "Sex", "Avg.Fare")
rpart.train.1 <- train.complete[1:891, features]

rpart.1.cv.3 <- rpart.cv(0123, rpart.train.1, rf.label, ctrl.4)
rpart.1.cv.3

prp(rpart.1.cv.3$finalModel, type = 0, extra = 4, under = TRUE)

sum(is.na(train.complete$Age))

# Work on titles
library(stringr)
name.splits <- str_split(train.complete$Name, ",")
name.splits <- str_split(sapply(name.splits, "[", 2), " ")
titles <- sapply(name.splits, "[", 2)
unique(titles)

train.complete[which(titles == "the"), ]

titles[titles %in% c("Dona.", "the")] <- "Lady."
titles[titles %in% c("Ms.", "Mlle.")] <- "Miss."
titles[titles == "Mme."] <- "Mrs."
titles[titles %in% c("Jonkheer.", "Don.")] <- "Sir."
titles[titles %in% c("Col.", "Capt.", "Major.")] <- "Officer"
table(titles)

train.complete$title <- as.factor(titles)

ggplot(train.complete[1:891,], aes(x = title, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Pclass) +
  scale_y_continuous(seq(0, 1, 0.1)) +
  ggtitle("Survival rates for title by pclass")

indexes <- which(train.complete$title == "Lady.")
train.complete$title[indexes] <- "Mrs."

indexes <- which(train.complete$title == "Dr." | train.complete$title == "Rev." |
                   train.complete$title == "Sir." | train.complete$title == "Officer")
train.complete$title[indexes] <- "Mr."

ggplot(train.complete[1:891,], aes(x = title, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Pclass) +
  scale_y_continuous(seq(0, 1, 0.1)) +
  ggtitle("Survival rates for compressed titles by pclass")

# Try a new part tree and see how titles look as a feature
features <- c("Pclass", "Avg.Fare", "title")
rpart.train.01 <- train.complete[1:891, features]

rpart.1.cv.2 <- rpart.cv(01234, rpart.train.01, rf.label, ctrl.4)
rpart.1.cv.2

prp(rpart.1.cv.2$finalModel, type = 0, extra = 4, under = TRUE)

# Still need to improve Mr title, too blunt
indexes.first.mr <- which(train.complete$title == "Mr." & train.complete$Pclass == "1")
first.mr.df <- train.complete[indexes.first.mr,]
summary(first.mr.df)

# Correct wrong index of female with Mr title
indexes <- which(train.complete$title == "Mr." & train.complete$Sex == "female")
train.complete$title[indexes] <- "Mrs."

indexes.first.mr <- which(train.complete$title == "Mr." & train.complete$Pclass == "1")
first.mr.df <- train.complete[indexes.first.mr,]

summary(first.mr.df[first.mr.df$Survived == "1",])
View(first.mr.df[first.mr.df$Survived == "1",])

ggplot(first.mr.df[first.mr.df$Survived != "None",], aes(x = Avg.Fare, fill = Survived)) +
         geom_density(alpha = 0.5) +
         ggtitle("1st Class Mr survival by average fare")

summary(train.complete$Avg.Fare)

# Do some preprocessing av the data, i.e. normalize the data
preproc.train.complete <- train.complete[, c("Group.Count", "Avg.Fare")]
preProc <- preProcess(preproc.train.complete, method = c("center", "scale"))

postproc.train.complete <- predict(preProc, preproc.train.complete)

# Check the correlation of group count and average fare -> not correlated
cor(postproc.train.complete$Group.Count, postproc.train.complete$Avg.Fare)

# Try new training with these new features
features <- c("Pclass", "title", "Group.Count", "Avg.Fare")
rpart.train.02 <- train.complete[1:891, features]

rpart.1.cv.4 <- rpart.cv(12345, rpart.train.02, rf.label, ctrl.4)
rpart.1.cv.4

prp(rpart.1.cv.4$finalModel, type = 0, extra = 4, under = TRUE, yesno = 2)

# Try new submission
test.submit.df <- train.complete[892:1309, c("Pclass", "title", "Group.Count", "Avg.Fare")]
rpart.3.preds <- predict(rpart.1.cv.4$finalModel, test.submit.df, type = "class")
table(rpart.3.preds)

submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rpart.3.preds)
write.csv(submit.df, file = "RF_SUB_20170409_1.csv", row.names = FALSE)

# Try submission with RandomForest
features <- c("Pclass", "title", "Group.Count", "Avg.Fare")
rf.train.temp <- train.complete[1:891, features]

set.seed(1234)
rf.temp <- randomForest(x = rf.train.temp, y = rf.label, ntree = 1000)
rf.temp

test.submit.df <- train.complete[892:1309, features]

rf.preds <- predict(rf.temp, test.submit.df)
table(rf.preds)

submit.df <- data.frame(PassengerId = rep(892:1309), Survived = rf.preds)
write.csv(submit.df, file = "RF_SUB_20170409_2.csv", row.names = FALSE)

# Represent the decision tree in a graphic format for females and boys
library(Rtsne)
most.correct <- train.complete[train.complete$title != "Mr.",]
indexes <- which(most.correct$Survived != "None")

tsne.1 <- Rtsne(most.correct[, features], check_duplicates = FALSE)
ggplot(NULL, aes(x = tsne.1$Y[indexes, 1], y = tsne.1$Y[indexes, 2], color = most.correct$Survived[indexes])) +
  geom_point() +
  labs(color = "Survived") +
  ggtitle("tsne 2D visualisation for females and boys")

# Check the conditional mutual information
library(infotheo)
condinformation(most.correct$Survived[indexes], discretize(tsne.1$Y[indexes,]))

condinformation(rf.label, train.complete[1:891, c("title", "Pclass")])

# Represent the decision tree in a graphic format for adult males
misters <- train.complete[train.complete$title == "Mr.",]
indexes <- which(misters$Survived != "None")

tsne.2 <- Rtsne(misters[, features], check_duplicates = FALSE)
ggplot(NULL, aes(x = tsne.2$Y[indexes, 1], y = tsne.2$Y[indexes, 2], color = misters$Survived[indexes])) +
  geom_point() +
  labs(color = "Survived") +
  ggtitle("tsne 2D visualisation for females and boys")

# Check the conditional mutual information
condinformation(misters$Survived[indexes], discretize(tsne.2$Y[indexes,]))
