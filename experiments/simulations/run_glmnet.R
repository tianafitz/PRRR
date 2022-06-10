library(magrittr)

dir <- "~/Documents/beehive/rrr/PRRR/experiments/simulations/tmp"
X_train <- read.csv(file.path(dir, "X_train.csv")) %>% as.matrix()
X_test <- read.csv(file.path(dir, "X_test.csv")) %>% as.matrix()
Y_train <- read.csv(file.path(dir, "Y_train.csv")) %>% as.matrix()

fit <- glmnet::glmnet(x = X_train, y = log(Y_train + 1), family = "mgaussian", lambda = 1)
preds <- predict(object = fit, newx = X_test, s = 1)
write.csv(x = preds, file = file.path(dir, "glmnet_preds.csv"))
