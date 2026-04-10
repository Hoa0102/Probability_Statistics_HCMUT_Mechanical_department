# --- 1. DATA PREPARATION ---
setwd(".")
data <- read.csv("data.csv")

# Inspect data structure and integrity
str(data)

print(any(is.na(data)))
print(sum(duplicated(data)))


# --- 2. PACKAGES & LIBRARIES ---
options(repos = c(CRAN = "https://cran.r-project.org"))
if (!require("pacman")) install.packages("pacman")
library(pacman)

# Load required analysis and modeling packages
p_load(tidyr, tidyverse, ggcorrplot, patchwork, caret, randomForest)
library(tidyr)        # Tidies and reshapes data efficiently.
library(tidyverse)    # Data manipulation & visualization suite (dplyr, ggplot2,...).
library(ggcorrplot)   # Visualizes correlation matrices with heatmaps.
library(patchwork)    # Combines ggplot2 plots into a single display.
library(caret)        # Data preprocessing and model training
library(randomForest) # Random forest algorithm implementation

# --- 3. DATA PREPROCESSING ---
# Encode categorical variables as numeric for modeling
temp_data <- data %>%
  mutate(infill_pattern = as.numeric(infill_pattern != "grid"),
         material = as.numeric(material != "abs"))
gathered_data <- gather(temp_data)
# Verify transformation
str(temp_data[c("infill_pattern", "material")])
summary(temp_data)

# --- 4. DATA VISUALIZATION ---
gathered_data <- gathered_data %>%
  mutate(key = factor(key, levels = unique(gathered_data$key)))

# Distribution analysis (Histograms)
ggplot(data = gathered_data, aes(x = value)) + 
  geom_histogram() +
  facet_wrap(~ key, scales = "free", nrow = 4) +
  labs(title = "Feature Distributions", x = NULL, y = NULL)

# Outlier detection (Box plots)
gathered_data_filtered <- gathered_data %>%
  filter(!key %in% c("infill_pattern", "material"))

ggplot(data = gathered_data_filtered, aes(y = as.numeric(value))) +
  stat_boxplot(geom = "errorbar", width = 0.2) +
  geom_boxplot(outlier.shape = NA) +
  geom_point(aes(x = -0.75), 
    position = position_jitter(width = 0.25, height = 0)) +
  facet_wrap(~ key, scales = "free", nrow = 2) +
  labs(x = NULL, y = NULL) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        plot.title = element_text(hjust = 0.5))

# Feature correlation matrix
cor = round(cor(temp_data), 2)

ggcorrplot(cor,
  method = "square", type = "upper", 
  lab = TRUE, lab_size = 3.5,
  title = "Correlation Matrix",
  legend.title = "Pearson\nCorrelation\n")

# Scatter plots for key mechanical properties
p1 <- ggplot(temp_data, aes(x = layer_height, y = roughness, color = factor(material))) + geom_point(size = 3)
p2 <- ggplot(temp_data, aes(x = fan_speed, y = tension_strenght, color = factor(material))) + geom_point(size = 3)
p3 <- ggplot(temp_data, aes(x = infill_pattern, y = elongation, color = factor(material))) + 
  xlim(-0.5, 1.5) + geom_point(size = 3)
(p1 | p2) / p3

# --- 5. MODELING SETUP ---
# Helper function to calculate NMAE and R-squared
calculate_metrics <- function(model, train_df, test_df, target) {
  pred_train <- predict(model, newdata = train_df)
  pred_test <- predict(model, newdata = test_df)
  
  actual_test <- test_df[[target]]
  range_val <- max(train_df[[target]]) - min(train_df[[target]])
  
  nmae_train <- mean(abs(pred_train - train_df[[target]])) / range_val
  nmae_test <- mean(abs(pred_test - actual_test)) / range_val
  
  rsq <- 1 - (sum((pred_test - actual_test)^2) / sum((actual_test - mean(actual_test))^2))
  
  cat("\nTarget:", target, "\n")
  cat("NMAE Train:", round(nmae_train, 7), "\n")
  cat("NMAE Test :", round(nmae_test, 7), "\n")
  cat("R-squared :", round(rsq, 7), "\n")
}

set.seed(75)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
train_idx <- sample(1:nrow(temp_data), 0.8 * nrow(temp_data))
train_all <- temp_data[train_idx, ]
test_all <- temp_data[-train_idx, ]

# --- 6. RANDOM FOREST REGRESSION ---
# Target: Roughness
rf_rough <- train(roughness ~ ., data = train_all[, !names(train_all) %in% c("tension_strenght", "elongation")], 
                  method = "rf", trControl = control)
calculate_metrics(rf_rough, train_all, test_all, "roughness")

# Target: Tensile Strength
rf_tension <- train(tension_strenght ~ ., data = train_all[, !names(train_all) %in% c("roughness", "elongation")], 
                    method = "rf", trControl = control)
calculate_metrics(rf_tension, train_all, test_all, "tension_strenght")

# Target: Elongation
rf_elong <- train(elongation ~ ., data = train_all[, !names(train_all) %in% c("roughness", "tension_strenght")], 
                  method = "rf", trControl = control)
calculate_metrics(rf_elong, train_all, test_all, "elongation")

# --- 7. ELASTIC NET REGRESSION ---
en_grid <- expand.grid(alpha = 0.5, lambda = seq(0.001, 1, length = 100))

# Target: Roughness
en_rough <- train(roughness ~ ., data = train_all[, !names(train_all) %in% c("tension_strenght", "elongation")], 
                  method = "glmnet", trControl = control, tuneGrid = en_grid)
calculate_metrics(en_rough, train_all, test_all, "roughness")

# Target: Tensile Strength
en_tension <- train(tension_strenght ~ ., data = train_all[, !names(train_all) %in% c("roughness", "elongation")], 
                    method = "glmnet", trControl = control, tuneGrid = en_grid)
calculate_metrics(en_tension, train_all, test_all, "tension_strenght")

# Target: Elongation
en_elong <- train(elongation ~ ., data = train_all[, !names(train_all) %in% c("roughness", "tension_strenght")], 
                  method = "glmnet", trControl = control, tuneGrid = en_grid)
calculate_metrics(en_elong, train_all, test_all, "elongation")