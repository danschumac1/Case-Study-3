# Illustration of deicision trees 
# Use the tree package
# This is an example exercise taken from the book “An Introduction to Statistical Learning with Applications in R” 
# by Gareth James, Deniela Witten, Trever Hastie.
################################################################### 
# Fitting a classification tree -->
################################################################### 
# install.packages("tree")
# install.packages("ISLR")


library(tree)
library(ISLR)

# Have a look at the data. 
data("Carseats")
head(Carseats)

# See the data structure
str(Carseats)
Carseats = Carseats
# Create a binary response, since sales is coninuous. 
# I chose the mean (7.5) as the separator. 
High <- ifelse(Carseats$Sales <= 7.5, "No","Yes")
Carseats <- data.frame(Carseats,High)
Carseats$High = as.factor(Carseats$High)

# Grow a tree
tree.carseats = tree(High ~ . - Sales, data = Carseats)

# See the default output 
tree.carseats

# See a summary of results
summary(tree.carseats)

# Plot the tree 
plot(tree.carseats)
text(tree.carseats, cex = 0.7)

#Estimate test error rate using validation set approach -->
set.seed(123)
train = sample(1:nrow(Carseats), 200)
# train and test
carseats.train = Carseats[train,]
carseats.test = Carseats[-train, ]

# Grow a tree using the training set 
tree2.carseats = tree(High ~ . - Sales, data = carseats.train)


# Get predictions on the test set
preds = predict(tree2.carseats, newdata = carseats.test, type="class")

#Compute the confusion matrix 
table(preds, carseats.test$High)

# Perform cost complexity pruning by cross-validation (CV), using misclassification rate
set.seed(123)
cv.carseats = cv.tree(tree2.carseats, FUN = prune.misclass)

# Note: k = alpha (pruning), dev = cross-validation error rate, size = size of tree -->

# Look at what is stored in the result object -->
cv.carseats
# Plot the estimated test error rate
par(mfrow = c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

# Note that when k is small, pruning is small, size is large, and vice versa. 
# Get the best size 
best_size = cv.carseats$size[which.min(cv.carseats$dev)]
best_size #16 node tree is the best size

#Get the pruned tree of the best size 
prune.carseats = prune.misclass(tree2.carseats, best = best_size)

# Plot the pruned tree. Nine leaves.
plot(prune.carseats)
text(prune.carseats,cex = 0.7)

# Get predictions on the test set
preds_pruned = predict(prune.carseats, newdata = carseats.test, type = "class")

# Get the confusion matrix  
table(preds_pruned, carseats.test$High)

# Compute the missclassification rate of a larger pruned tree for size 15.


# You can see the accuracy decreases on the testing data, as we use a larger size tree (alpha = k smaller)

################################################################### -->
# Fitting a regression tree -->
################################################################### 

library(MASS)

data(Boston)
head(Boston)
# We will use the housing data from Boston. 
str(Boston)

# Create training and test sets. 50/50 split.
set.seed(123)
train = sample(1:nrow(Boston),200)
Boston.train = Boston[train, ]
Boston.test = Boston[-train, ]

# Grow a tree using the training set. Media value (medv) is the response. 
tree.boston = tree(medv ~ ., data = Boston.train)
summary(tree.boston)
# Plot the tree 
plot(tree.boston)
text(tree.boston)

# Perform cost complexity pruning by CV
cv.boston = cv.tree(tree.boston)
cv.boston
best_size = cv.boston$size[which.min(cv.boston$dev)]
# Plot the estimated test error rate 
plot(cv.boston$size, cv.boston$dev, type = "b")

# Note: Best size = 8 (i.e., no pruning) 
  
# If needed, pruning can be performed by specifying the "best" argument 

prune.boston = prune.tree(tree.boston,best=5) #prune.tree for regression, prune.misclass for class


# Get predictions on the test data 
preds_bos_pruned = predict(prune.boston, newdata=Boston.test)
preds_bos = predict(tree.boston, newdata = Boston.test)

# Plot the observed values against the predicted values
plot(prune.boston)
text(prune.boston)
# Compute the test error rate 
mean((preds_bos_pruned - Boston.test$medv)^2)
mean((preds_bos - Boston.test$medv)^2)

