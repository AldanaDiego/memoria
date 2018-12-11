## Benchmark de algoritmos de clasificacion para variable L
## Se utilizan los datos originales sin reducir
## Se prueba con cross validation
## Algoritmos utilizados
# 1. SVM Radial
# 2. SVM Lineal
# 3. KNN
# 4. Naive Bayes
# 5. LDA
# 6. CART
# 7. C4.5
# 8. Random Forest

methods <- c("SVMR","SVML","KNN","NB","LDA","CART","C45","RF")
number_methods <- length(methods)

library(caret) #knn, confusion matrix
library(e1071) #svm, naive bayes 
library(MASS) #LDA
library(rpart) #CART
library(RWeka) #C4.5
library(randomForest) #Random forest XD

data.train <- read.csv("CE_L_0.01_10_01.csv")
data.test <- read.csv("CP_L_0.01_10_01.csv")
names(data.train) <- c(1:5000)
names(data.test) <- c(1:5000)

#Juntar sets y quitar columna de clase
data.all <- rbind(data.train, data.test)
label.all <- data.all[,ncol(data.all)]
data.all <- data.all[,-ncol(data.all)]

#Generar indices desordenados para sets de entrenamiento y prueba
indexes <- sample(x = nrow(data.all), size = nrow(data.all), replace = FALSE)

#Cross Validation
results <- as.data.frame(matrix(data = NA, nrow=number_methods, ncol=10))
row.names(results) <- methods

for (i in 1:10) {
    
  #Obtener el i-esimo grupo como set de prueba
  start <- (i-1)*floor(nrow(data.all)/10) + 1
  end <- i*floor(nrow(data.all)/10)
  test_index <- indexes[start:end]
  
  #Separar datos de entrenamiento y prueba
  data.train <- data.all[-test_index,]
  data.test <- data.all[test_index,]
  label.train <- as.factor(label.all[-test_index])
  label.test <- as.factor(label.all[test_index])
  
  
  #SVM Radial
  # print("SVM Radial")
  # tuned_model <- tune(train.x = data.train, train.y = label.train, method = svm, kernel = "radial",
  #                    ranges=list(cost=10^(-2:2), gamma=c(.5,1,2,.001*(1:5))))
  # model <- tune_model$best.model
  model <- svm(label.train ~ ., data=data.train, kernel="radial")
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["SVMR",i] <- cmat$overall[1]
  
  
  #SVM Lineal
  # print("SVM Lineal")
  # tuned_model <- tune(train.x = data.train, train.y = label.train, method = svm, kernel = "linear",
  #                    ranges=list(cost=10^(-2:2), gamma=c(.5,1,2,.001*(1:5))))
  # model <- tune_model$best.model
  model <- svm(label.train ~ ., data=data.train, kernel="linear")
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["SVML",i] <- cmat$overall[1]
  
  
  #KNN
  # print("KNN")
  # tuned_model <- tune.knn(x = data.train, y = label.train, k = c(3:10))
  # model <- knn3(x = data.train, y = label.train, k = tuned_model$best.parameters$k)
  model <- knn3(x = data.train, y = label.train, k = floor( sqrt( nrow(data.train) ) ))
  predicted <- predict(model, data.test, type = "class")
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["KNN",i] <- cmat$overall[1]
  
  
  #Naive Bayes
  # print("Naive Bayes")
  model <- naiveBayes(x = data.train, y = label.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["NB",i] <- cmat$overall[1]
  
  
  #LDA
  # print("LDA")
  model <- lda(as.vector(label.train), x = data.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=as.factor(predicted$class), reference=label.test)
  results["LDA",i] <- cmat$overall[1]

  
  #CART
  # print("CART")
  # tuned_model <- train(x = data.train, y = label.train, method="rpart", tuneGrid = expand.grid(cp = c(1) ))
  # model <- tuned_model$finalModel
  model <- rpart(label.train ~ ., data = data.train)
  predicted <- predict(model, data.test, type="class")
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["CART",i] <- cmat$overall[1]

    
  #C4.5
  # print("C4.5")
  # tuned_model <- train(x = data.train, y = label.train, method = "J48", tuneGrid = expand.grid(C = c(0.1*(1:5)), M = c(1:5) ))
  # model <- tuned_model$finalModel
  model <- J48(label.train ~ ., data = data.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["C45",i] <- cmat$overall[1]
  
  
  #Random Forest
  # print("Random Forest")
  # tuned_model <- train(x = data.train, y = label.train, method="rf", tuneGrid = expand.grid(.mtry = c(1:15) ))
  # model <- tuned_model$finalModel
  model <- randomForest(x = data.train, y = label.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["RF",i] <- cmat$overall[1]
  
}

write.csv(results, file = "benchmark_L.out")

library(ggplot2)
library(reshape2)
ggplot(melt(as.matrix(results)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "Precisión", breaks = seq(0.6, 1, 0.05), limits = c(0.6, 1)) + 
  ggtitle("Precisión obtenida para L con datos sin reducir")