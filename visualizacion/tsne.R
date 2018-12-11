## Usar t-SNE para obtener graficos en 2D de los datos
## Se usan los datos de clasificacion y regresion

library(ggplot2)
library(Rtsne)

#Datos para clasificacion
#https://drive.google.com/open?id=1by4cqw7vH5hAorH_Ve6oUCFRw5dksjzl
id <- "1by4cqw7vH5hAorH_Ve6oUCFRw5dksjzl"
data<-read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", id))

#Obtenemos las variables R y L para "pintar" los graficos
R <- as.factor(data[,"R"])
L <- as.factor(data[,"L"])
data <- data[,-c(5001:5002)]

#Calcular la representacion en 2D de los datos con t-SNE
#Este sera el dataframe a graficar
tsne <- Rtsne(data)

#Grafico para clasificacion de variable R
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2])
ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=R))
ggsave("tsne_clasif_R.png")

#Grafico para clasificacion de variable L
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2])
ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=L))
ggsave("tsne_clasif_L.png")


#Datos para regresion
#https://drive.google.com/open?id=1Dukxv-ueabowsKPMgOcF4ssdbCa3Do9q
id <- "1Dukxv-ueabowsKPMgOcF4ssdbCa3Do9q"
data<-read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", id))

#Obtenemos las variables R y L para "pintar" los graficos
#Seguiremos usando el t-SNE calculado anteriormente, solo necesitamos las
# variables R y L de este dataset
R <- data[,"R"]
L <- data[,"L"]

#Grafico para regresion de variable R
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2])
ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=R)) + scale_color_gradient(low="blue", high = "red")
ggsave("tsne_regres_R.png")

#Grafico para regresion de variable L
tsne_plot <- data.frame(x = tsne$Y[,1], y = tsne$Y[,2])
ggplot(tsne_plot) + geom_point(aes(x=x, y=y, color=L)) + scale_color_gradient(low="blue", high = "red")
ggsave("tsne_regres_L.png")