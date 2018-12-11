## Benchmark de algoritmos de clasificacion para variable R
## Se utilizan los datos reducidos mediante algoritmo Boruta
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
library(Boruta)

data.train <- read.csv("CE_R_10_10_01.csv")
data.test <- read.csv("CP_R_10_10_01.csv")
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
  
  #Reducir datos eliminando correlaciones y usando Boruta
  high_correlation <- findCorrelation(cor(data.train), cutoff = 0.95)
  selected_features <- c(1:ncol(data.train))[-high_correlation]
  model <- Boruta(y = label.train, x = data.train[,selected_features])
  selected_features <- getSelectedAttributes(model, withTentative = FALSE)
  data.train <- data.train[,selected_features]
  data.test <- data.test[,selected_features]
  
  
  #SVM Radial
  # print("SVM Radial")
  model <- svm(label.train ~ ., data=data.train, kernel="radial")
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["SVMR",i] <- cmat$overall[1]
  
  #SVM Lineal
  # print("SVM Lineal")
  model <- svm(label.train ~ ., data=data.train, kernel="linear")
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["SVML",i] <- cmat$overall[1]
  
  #KNN
  # print("KNN")
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
  model <- rpart(label.train ~ ., data = data.train)
  predicted <- predict(model, data.test, type="class")
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["CART",i] <- cmat$overall[1]
  
  #C4.5
  # print("C4.5")
  model <- J48(label.train ~ ., data = data.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["C45",i] <- cmat$overall[1]
  
  #Random Forest
  # print("Random Forest")
  model <- randomForest(x = data.train, y = label.train)
  predicted <- predict(model, data.test)
  cmat <- confusionMatrix(data=predicted, reference=label.test)
  results["RF",i] <- cmat$overall[1]
  
}

write.csv(results, file = "benchmark_R_boruta.out")

library(ggplot2)
library(reshape2)
ggplot(melt(as.matrix(results)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "Precisión", breaks = seq(0.45, 1, 0.05), limits = c(0.45, 1)) + 
  ggtitle("Precisión obtenida para R usando Boruta")