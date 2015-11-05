directory <- "/home/victor/desktop/mlearning/unicamp/assignment4/"

number_of_features <- 300

df <- read.csv(paste(directory, "dataset_ok.txt", sep=""), header=F)
features <- sample(4:ncol(df), number_of_features)

x_indexes <- seq(from = 4, to = 100*3+1, by = 3)
y_indexes <- seq(from = 5, to = 100*3+1, by = 3)
z_indexes <- seq(from = 6, to = 100*3+1, by = 3)

df[, x_indexes] <- apply(df[, x_indexes], 1, fft)
df[, y_indexes] <- apply(df[, y_indexes], 1, fft)
df[, z_indexes] <- apply(df[, z_indexes], 1, fft)

df[, x_indexes] <- apply(df[, x_indexes], 1, Mod)
df[, y_indexes] <- apply(df[, y_indexes], 1, Mod)
df[, z_indexes] <- apply(df[, z_indexes], 1, Mod)

# Normalizing
# Row-wise
df$mean_row_X <- apply(df[, x_indexes], 1, mean)
df$mean_row_Y <- apply(df[, y_indexes], 1, mean)
df$mean_row_Z <- apply(df[, z_indexes], 1, mean)

df$sd_row_X <- apply(df[, x_indexes], 1, sd)
df$sd_row_Y <- apply(df[, y_indexes], 1, sd)
df$sd_row_Z <- apply(df[, z_indexes], 1, sd)

df[, x_indexes] <- df[, x_indexes] - df$mean_row_X
df[, y_indexes] <- df[, y_indexes] - df$mean_row_Y
df[, z_indexes] <- df[, z_indexes] - df$mean_row_Z

df[, x_indexes] <- df[, x_indexes] / df$sd_row_X
df[, y_indexes] <- df[, y_indexes] / df$sd_row_Y
df[, z_indexes] <- df[, z_indexes] / df$sd_row_Z

# General way normalization
# df$mean <- apply(df[, 4:ncol(df)], 1, mean)
# df$sd <- apply(df[, 4:ncol(df)], 1, sd)
# df[, 4:ncol(df)] <- df[, 4:ncol(df)] - df$mean
# df[, 4:ncol(df)] <- df[, 4:ncol(df)] / df$sd

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
    cat("\n\n")
    print(class)

    training$Activity <- 0
    training$Activity[training$V2 == class] <- 1

    model <- glm(form, data=training, family=binomial())
    testing[paste("predict_", i)] <- predict(model, testing, type="response")
    
    i <- i+1
    # Cleaning the memory
    model <- 0
}

# Running the tests
# talvez chamar which.max sem apply
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

cat("\n\n\n")
# CROSS VALIDATION
# Training the models
for (class in classes) {
    cat("\n\n")
    print(class)

    testing$Activity <- 0
    testing$Activity[testing$V2 == class] <- 1

    model <- glm(form, data=testing, family=binomial())
    training[paste("predict_", i)] <- predict(model, training, type="response")
    
    i <- i+1
    # Cleaning the memory
    model <- 0
}

# Running the tests
# talvez chamar which.max sem apply
training$class <- apply(training[, c(grep("predict_", colnames(training)))], 1, which.max)
training$classname <- classes[training$class]

for (class in classes) {
  cat("\n\n\n")
  print(class)
  subset <- training[training$V2 == class, ]
  print(paste("total rows: ", nrow(subset)))
  
  for (class2 in classes) {
    cat("\n")
    print(paste(class2, ":", nrow(subset[subset$classname == class2, ])))
  }
}
