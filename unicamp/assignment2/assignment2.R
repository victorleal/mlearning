rm(list=ls())
#time1 <- proc.time()

#Load the data
df <- read.csv(file.choose(), header=T)

df$datetime <- strptime(x = as.character(df$Dates), format="%Y-%m-%d %H:%M:%S")

#https://stat.ethz.ch/R-manual/R-devel/library/base/html/DateTimeClasses.html
# df$hour <- as.POSIXlt(df$datetime)$hour
# df$minute <- as.POSIXlt(df$datetime)$min
# df$second <- as.POSIXlt(df$datetime)$sec
# df$day <- as.POSIXlt(df$datetime)$mday
# df$month <- as.POSIXlt(df$datetime)$mon+1
# df$year <- as.POSIXlt(df$datetime)$year+1900
# df$wday <- as.POSIXlt(df$datetime)$wday

#http://stackoverflow.com/questions/23103223/converting-factors-to-numeric-values-in-r
df$PdDistrict.f <- as.numeric(factor(df$PdDistrict , levels(df$PdDistrict)))
#df$PdDistrict <- factor(df$PdDistrict)

#drops <- c('Dates', 'DayOfWeek', 'Resolution', 'Address', 'Descript', 'datetime')
#data <- df[, !(names(df) %in% drops)]
#head(data)
categories = levels(df$Category)

# Transform the days of week on binary features
days_of_week <- levels(df$DayOfWeek)
for(i in 1:length(days_of_week)) {
  df$days_of_week[i] <- 0
}

training = df[1:700000,]

for(i in 1:length(categories)) {
  cat("\n\n")
  time1 <- proc.time()
  
  print(categories[i])
  training$Category.W <- 0
  training$Category.W[training$Category==categories[i]] <- 1 
  
  mylogit <- glm(Category.W ~ X+Y+PdDistrict.f, data=training, family=binomial(logit))
  print(summary(mylogit))
  
  time2 <- proc.time()
  print(time2 - time1)
}

