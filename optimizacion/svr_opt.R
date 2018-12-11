##Regresion de variables R y L usando Support Vector Regression
##Se realiza una busqueda de parametros para cost y gamma
##Se reducen los datos usando correlaciones
##No se realizo 10FCV
##Se uso 10% de los datos como prueba

library(caret)
library(e1071)
library(ggplot2)
library(Metrics)

data <- read.csv("data_regression.csv")

label.R <- data[,"R"]
label.L <- data[,"L"]
data <- data[, -c((ncol(data)-1):(ncol(data)))]

#Reducir dimension eliminando columnas con alta correlacion
zv <- apply(data, 2, function(x) length(unique(x)) == 1)
data <- data[,-zv]
high_correlation <- findCorrelation(cor(data), cutoff = 0.95)
selected_features <- c(1:ncol(data))[-high_correlation]
data <- data[,selected_features]

#Tomar 10% como datos de prueba
test <- sample(x = nrow(data), size = nrow(data)/10, replace = FALSE)

##Regresion para R
#model.R <- svm(x = data[-test,], y = label.R[-test])
tuned.R <- tune(svm, train.x = data[-test,], train.y = label.R[-test], ranges = list(cost = 2^(-2:2), gamma = 2^(-2:2), kernel = c("radial", "linear")))
model.R <- tuned.R$best.model
predicted.R <- predict(model.R, data[test,])

#Calcular metricas
mae.R <- mae(actual = label.R[test], predicted = predicted.R)
mse.R <- mse(actual = label.R[test], predicted = predicted.R)

#Graficar resultados
graph_label.R <- data.frame(x = c(1:length(test)), y = label.R[test], z = "Real value")
graph_predicted.R <- data.frame(x = c(1:length(test)), y = predicted.R, z = "Predicted")
graph.R <- rbind(graph_label.R, graph_predicted.R)
ggplot(graph.R, aes(x=x, y=y, group=z)) + geom_point(aes(color = z)) + geom_line(aes(color = z)) + labs(x = "Prueba", y = "R", colour = "Value") + ylim(c(8.5, 11.5))
ggsave("svr_opt_R2.png")


##Regresion para L
#model.L <- svm(x = data[-test,], y = label.L[-test])
tuned.L <- tune(svm, train.x = data[-test,], train.y = label.L[-test], ranges = list(cost = 2^(-2:2), gamma = 2^(-2:2), kernel = c("radial", "linear")))
model.L <- tuned.L$best.model
predicted.L <- predict(model.L, data[test,])

#Calcular metricas
mae.L <- mae(actual = label.L[test], predicted = predicted.L)
mse.L <- mse(actual = label.L[test], predicted = predicted.L)

#Graficar resultados
graph_label.L <- data.frame(x = c(1:length(test)), y = label.L[test], z = "Real value")
graph_predicted.L <- data.frame(x = c(1:length(test)), y = predicted.L, z = "Predicted")
graph.L <- rbind(graph_label.L, graph_predicted.L)
ggplot(graph.L, aes(x=x, y=y, group=z)) + geom_point(aes(color = z)) + geom_line(aes(color = z)) + labs(x = "Prueba", y = "L", colour = "Value")
ggsave("svr_opt_L2.png")
