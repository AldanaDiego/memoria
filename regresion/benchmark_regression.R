##Regresion de variables R y L usando varios algoritmos
##Se reducen los datos usando correlaciones
##Probar usando cross validation
##Regresores utilizados:
##1. Support Vector Regression
##2. Multiple Linear Regression
##3. Principal Component Regression 
##4. CART
##5. Random Forest

library(caret)
library(e1071)
library(ggplot2)
library(Metrics) #Metricas de evaluacion
library(reshape2) #Procesa datos para el grafico
library(pls) #PCR
library(rpart) #CART
library(randomForest)

data <- read.csv("data_regression.csv")

label.R <- data[,"R"]
label.L <- data[,"L"]
data <- data[, -c((ncol(data)-1):(ncol(data)))]

#Generar indices desordenados para sets de entrenamiento y prueba
indexes <- sample(x = nrow(data), size = nrow(data), replace = FALSE)

#Arreglos para metricas de evaluacion
methods <- c("SVR","MLR","PCR", "CART","RF")
number_methods <- length(methods)

mae.R <- as.data.frame(matrix(data = NA, nrow=number_methods, ncol=10))
mae.L <- as.data.frame(matrix(data = NA, nrow=number_methods, ncol=10))
mse.R <- as.data.frame(matrix(data = NA, nrow=number_methods, ncol=10))
mse.L <- as.data.frame(matrix(data = NA, nrow=number_methods, ncol=10))
row.names(mae.R) <- methods
row.names(mae.L) <- methods
row.names(mse.R) <- methods
row.names(mse.L) <- methods

for (i in 1:10) {
  #Obtener el i-esimo grupo como set de prueba
  start <- (i-1)*floor(nrow(data)/10) + 1
  end <- i*floor(nrow(data)/10)
  test_index <- indexes[start:end]
  
  #Separar datos de entrenamiento y prueba
  data.train <- data[-test_index,]
  data.test <- data[test_index,]
  
  #Reducir datos eliminando columnas con alta correlacion
  zv <- apply(data.train, 2, function(x) length(unique(x)) == 1)
  data.train <- data.train[,-zv]
  data.test <- data.test[,-zv]
  high_correlation <- findCorrelation(cor(data.train), cutoff = 0.95)
  selected_features <- c(1:ncol(data.train))[-high_correlation]
  data.train <- data.train[,selected_features]
  data.test <- data.test[,selected_features]
  
  ##SVR para R
  model.R <- svm(label.R[-test_index] ~., data.train)#, cost = 0.25, gamma = 0.25, kernel = "linear")
  predicted.R <- predict(model.R, data.test)
  mae.R["SVR", i] <- mae(actual = label.R[test_index], predicted = predicted.R)
  mse.R["SVR", i] <- mse(actual = label.R[test_index], predicted = predicted.R)
  
  ##SVR para L
  model.L <- svm(label.L[-test_index] ~., data.train)#, cost = 0.25, gamma = 0.25, kernel = "linear")
  predicted.L <- predict(model.L, data.test)
  mae.L["SVR",i] <- mae(actual = label.L[test_index], predicted = predicted.L)
  mse.L["SVR",i] <- mse(actual = label.L[test_index], predicted = predicted.L)
  
  #MLR para R
  model.R <- lm(formula = label.R[-test_index] ~., data = data.train)
  predicted.R <- predict(model.R, data.test)
  mae.R["MLR", i] <- mae(actual = label.R[test_index], predicted = predicted.R)
  mse.R["MLR", i] <- mse(actual = label.R[test_index], predicted = predicted.R)
  
  #MLR para L
  model.L <- lm(formula = label.L[-test_index] ~., data = data.train)
  predicted.L <- predict(model.L, data.test)
  mae.L["MLR", i] <- mae(actual = label.L[test_index], predicted = predicted.L)
  mse.L["MLR", i] <- mse(actual = label.L[test_index], predicted = predicted.L)  
  
  #PCR para R
  model.R <- pcr(label.R[-test_index] ~., data = data.train)
  predicted.R <- predict(model.R, data.test)
  mae.R["PCR", i] <- mae(actual = label.R[test_index], predicted = predicted.R)
  mse.R["PCR", i] <- mse(actual = label.R[test_index], predicted = predicted.R)
  
  #PCR para L
  model.L <- pcr(label.L[-test_index] ~., data = data.train)
  predicted.L <- predict(model.L, data.test)
  mae.L["PCR", i] <- mae(actual = label.L[test_index], predicted = predicted.L)
  mse.L["PCR", i] <- mse(actual = label.L[test_index], predicted = predicted.L)
  
  #CART para R
  model.R <- rpart(label.R[-test_index] ~., data = data.train)
  predicted.R <- predict(model.R, data.test)
  mae.R["CART", i] <- mae(actual = label.R[test_index], predicted = predicted.R)
  mse.R["CART", i] <- mse(actual = label.R[test_index], predicted = predicted.R)
  
  #CART para L
  model.L <- rpart(label.L[-test_index] ~., data = data.train)
  predicted.L <- predict(model.L, data.test)
  mae.L["CART", i] <- mae(actual = label.L[test_index], predicted = predicted.L)
  mse.L["CART", i] <- mse(actual = label.L[test_index], predicted = predicted.L)
  
  #Random forest para R
  model.R <- randomForest(x = data.train, y = label.R[-test_index])
  predicted.R <- predict(model.R, data.test)
  mae.R["RF", i] <- mae(actual = label.R[test_index], predicted = predicted.R)
  mse.R["RF", i] <- mse(actual = label.R[test_index], predicted = predicted.R)
  
  #Random Forest para L
  model.L <- randomForest(x = data.train, y = label.L[-test_index])
  predicted.L <- predict(model.L, data.test)
  mae.L["RF", i] <- mae(actual = label.L[test_index], predicted = predicted.L)
  mse.L["RF", i] <- mse(actual = label.L[test_index], predicted = predicted.L)
}

#Graficar resultados
ggplot(melt(as.matrix(mae.R)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "MAE")
ggsave("benchmark_mae_R.png")

ggplot(melt(as.matrix(mse.R)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "MSE")
ggsave("benchmark_mse_R.png")

ggplot(melt(as.matrix(mae.L)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "MAE")
ggsave("benchmark_mae_L.png")

ggplot(melt(as.matrix(mse.L)), aes(x = Var1, y = value)) + 
  geom_boxplot(fill = "#3366FF", alpha = 0.7) + 
  scale_x_discrete(name = "Métodos") + 
  scale_y_continuous(name = "MSE")
ggsave("benchmark_mse_L.png")