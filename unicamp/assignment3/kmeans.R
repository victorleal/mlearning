#data <- read.csv(file.choose(), header=F)

#random <- sample(1:81432,10000)

#set <- data[random,]

df <- data.frame()

for (k in seq(500,3500, by=300)){
  cat("\nCurrent K:", k)
  km = kmeans(set, k)
  cat("\nInertia: ", km$tot.withinss)
  df <- rbind(df, c(k, km$tot.withinss))
}