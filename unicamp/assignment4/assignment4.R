directory <- "/home/victor/desktop/mlearning/unicamp/assignment4/"

number_of_features <- 150

df <- read.csv(paste(directory, "dataset_range_10.txt", sep=""), header=F)
features <- sample(4:ncol(df), number_of_features)

#df <- data.frame(df[1:3], apply(df[features], 2, fft))
#features <- colnames(df)
#features <- features[4:length(features)]

#df <- data.frame(df[1:3], apply(df[features], 2, as.numeric))

# data <- read.csv(paste(directory, "dataset_ok.txt", sep=""), header=F)
# 292218 e 146109 para range=10

training <- df[sample(1:nrow(df), floor(nrow(df)/2)),]
testing <- df[sample(1:nrow(df), floor(nrow(df)/2)),]

classes <- levels(df$V2)
rm(df)

models <- list()
i <- 1
form <- paste("Activity ~ ", paste(paste0("V", features), collapse= "+"))

# Training the models
for (class in classes) {
    #cat("\n\n")
    #print(class)

    model <- 0

    training$Activity <- 0
    training$Activity[training$V2 == class] <- 1

    model <- glm(form, data=training, family=binomial())
    testing[paste("predict_", i)] <- predict(model, testing, type="response")
    i <- i+1
}

# Running the tests
testing$class <- apply(testing[, c(grep("predict_", colnames(testing)))], 1, which.max)
testing$classname <- classes[testing$class]

for (class in classes) {
  cat("\n\n\n")
  print(class)
  subset <- testing[testing$V2 == class, ]
  print(paste("total rows: ", nrow(subset)))
  
  for (class2 in classes) {
    cat("\n")
    print(paste(class2, ":", nrow(subset[subset$classname == class2, ])))
  }
}

