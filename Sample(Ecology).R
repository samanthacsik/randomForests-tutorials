########################################
# Exercise 1: Sample(Ecology)
########################################

# CART: Classification and regression trees

library(rpart) # performs CART analysis (tends to be biased towards vars with many possible splits or many possible vals)

library(party) # another package that performs CART analysis (not as biases and also based on statisical stopping rules (e.g. p < 0.05))

# Random Forests: an ensemble of trees that are built using a variant of bagging (i.e. a bootstrapped sample of data to construct the model); only use a proportion of the predictors making trees truly independent from one another 

library(randomForest) 

########################################
# Build a simple classification tree using the `iris` dataset
# how do the morphological characters relating to sepals and petals relate predict the species of iris?
########################################

head(iris)

(tree.model <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris))

# first split is on Petal.Length with all 50 obs of I.setosa having a P.L < 2.45 cm
# further splits on Petal.Width: almost all I.versicolor have P.W < 1.75 cm AND almost all I.virginica have P.W >=1.75cm

# show graphically
plot(tree.model)
text(tree.model)

# investigate how these splits were made by exploring raw 

# first node
plot(iris$Petal.Length, col = c("red", "blue", "forestgreen")[iris$Species], xlab = "", ylab = "Petal Length (cm)")
legend(120, 2, c("setosa", "versicolor", "virginica"), col = c("red", "blue", "forestgreen"), pch = 1)
abline(h = 2.45)

# second node
plot(subset(iris, Species != "setosa")$Petal.Width, col = c("red", "blue", "forestgreen")[subset(iris, Species != "setosa")$Species], xlab = "", ylab = "Petal Width (cm)")
legend(80, 1.2, c("versicolor","virginica"), col = c("blue", "forestgreen"), pch = 1)
abline(h = 1.75)

# information on the pruning procedure and overall model fit using printcp()
printcp(tree.model)

# grow out the tree a bit more
tree.model.updated <- update(tree.model, control = rpart.control(minsplit = 2))

printcp(tree.model.updated)

# using party:
ctree.model <- ctree(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris)
plot(ctree.model)

########################################
# even with pruning, a single CART is likely to overfit the data
# one way to get around this si to build a bunch of trees on only subsets of the data, and generalize across them
# b/c any given tree is constructed with only a portion of the data, the likelihood of overfitting is drastically reduced; moreover, averaging across many trees is likely to wash out any spurious signals from a single tree (ensemble learning)
# "bagging" = taking a portion of the data and using it to build many trees
########################################

########################################
# now lets apply the random forest approach to the iris dataset
########################################

(RF.model <- randomForest(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = iris))

# reports number of trees in forest (500), number of predictors sampled at each split (2), OOB error rate (4% which is pretty low), and confusion matrix (number of misclassificatioms)

plot(RF.model)
