# --- 1. DATA PREPARATION ---
setwd(".")
data <- read.csv("data.csv", stringsAsFactors = FALSE)

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

# --- 3. DATA PREPROCESSING ---
# Encode categorical variables as numeric for modeling
colnames(data) <- trimws(gsub("[^[:alnum:]_]", "", colnames(data)))
temp_data <- data %>%
  mutate(infill_pattern = as.numeric(infill_pattern != "grid"),
         material = as.numeric(material != "abs")) %>%
  mutate(across(everything(), as.numeric))
# Verify transformation
str(temp_data[c("infill_pattern", "material")])


# --- 4. DATA VISUALIZATION ---
# Outlier detection (Box plots)
gathered_data <- temp_data %>%
  gather(key = "key", value = "value") %>%
  mutate(key = factor(key, levels = unique(key)))

gathered_data_filtered <- gathered_data %>%
  filter(key != "infill_pattern" & key != "material")

final_plot <- ggplot(data = gathered_data_filtered, aes(x = factor(1), y = as.numeric(value))) +
  stat_boxplot(geom = "errorbar", width = 0.2) +
  geom_boxplot(outlier.shape = NA, fill = "gold", width = 0.6) +
  geom_point(aes(x = 0.6),
             position = position_jitter(width = 0.1, height = 0),
             color = "red", alpha = 0.5, size = 2) +
    facet_wrap(~ key, scales = "free", nrow = 2) +
  labs(title = "Summary of Statistical Box Plots for Parameters", x = NULL, y = NULL) 
print(final_plot)

# Distribution analysis (Histograms)
gathered_data <- temp_data %>%
  gather(key = "key", value = "value") %>%
  mutate(key = factor(key, levels = unique(key)))

p_hist_combined <- ggplot(data = gathered_data, aes(x = value)) +
  geom_histogram(fill = "red", color = "black", bins = 15) +
  facet_wrap(~ key, scales = "free", ncol = 4) +
  labs(title = "Histograms", x = NULL, y = "Frequency")
print(p_hist_combined)

# Feature correlation matrix
cor_matrix <- round(cor(temp_data), 2)

p_cor <- ggcorrplot(cor_matrix,
        method = "square",# Hiện ô vuông giống báo cáo
        type = "upper",   # Chỉ hiện nửa tam giác trên để không bị rối
        lab = TRUE,       # Hiển thị các con số bên trong ô
        lab_size = 3.5,   # Chỉnh cỡ chữ số nhỏ lại để không bị tràn ô
        colors = c("purple", "white", "green"), # Thang màu từ tím (âm) sang xanh lá (dương)
        title = "Pearson Correlation Matrix",
        legend.title = "Pearson\nCorrelation\n") 
print(p_cor)

# Scatter plots for key mechanical properties
# 1. Elongation vs Tension Strength 
p_mech <- ggplot(temp_data, aes(x = elongation, y = tension_strenght, color = factor(material))) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Elongation vs Tension Strength")

# 2. Bed Temperature vs Fan Speed
p_bed_fan <- ggplot(temp_data, aes(x = bed_temperature, y = fan_speed)) +
  geom_point(size = 3, color = "steelblue", alpha = 0.8) +
  labs(title = "Bed Temp vs Fan Speed")
 # 3. Material vs Nozzle Temperature
p_mat_temp <- ggplot(temp_data, aes(x = factor(material, labels=c("ABS (0)", "PLA (1)")), y = nozzle_temperature, color = factor(material))) +
  geom_jitter(width = 0.15, size = 3, alpha = 0.8) +
  labs(title = "Material vs Nozzle Temp", x = "Material")
# 4. Layer Height vs Roughness
p_rough <- ggplot(temp_data, aes(x = layer_height, y = roughness, color = factor(material))) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Layer Height vs Roughness")
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

train_rough <- train_all[, !names(train_all) %in% c("tension_strenght", "elongation")]
test_rough  <- test_all[, !names(test_all) %in% c("tension_strenght", "elongation")]

rf_rough <- train(roughness ~ ., data = train_rough,
                method = "rf", trControl = control)

calculate_metrics(rf_rough, train_rough, test_rough, "roughness")

# Target: Tensile Strength

train_tension <- train_all[, !names(train_all) %in% c("roughness", "elongation")]
test_tension  <- test_all[, !names(test_all) %in% c("roughness", "elongation")]

rf_tension <- train(tension_strenght ~ ., data = train_tension, 
                method = "rf", trControl = control)

calculate_metrics(rf_tension, train_tension, test_tension, "tension_strenght")

# Target: Elongation

train_elong <- train_all[, !names(train_all) %in% c("roughness", "tension_strenght")]
test_elong  <- test_all[, !names(test_all) %in% c("roughness", "tension_strenght")]

rf_elong <- train(elongation ~ ., data = train_elong, 
                method = "rf", trControl = control)

calculate_metrics(rf_elong, train_elong, test_elong, "elongation")

calculate_metrics <- function(model, train_df, test_df, target){
  pred_train <- predict(model, newdata = train_df)
  pred_test <- predict(model, newdata = test_df)

  actual_test <- test_df[[target]]
  range_val <- max(train_df[[target]]) - min(train_df[[target]])

  nmae_train <- mean(abs(pred_train - train_df[[target]])) / range_val
  nmae_test <- mean(abs(pred_test - actual_test)) / range_val

  rsq <- 1 - (sum((pred_test - actual_test)^2) / sum((actual_test - mean(actual_test))^2))

  cat("\nTarget: ", target)
  cat("\nNMAE Train: ", round(nmae_train, 7))
  cat("\nNMAE Test: ", round(nmae_test, 7))
  cat("\nR-Squared: ", round(rsq, 7))
}
# --- 6. ELASTIC NET MODEL ---
library(caret)
library(glmnet)

train.rows <- sample(rownames(temp_data), 0.8 * nrow(temp_data))
test.rows  <- setdiff(rownames(temp_data), train.rows)

train.data <- temp_data[train.rows, ]
test.data  <- temp_data[test.rows, ]

# =========================
# 2. CV control
# =========================
control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5
)

# =========================
# 3. Function
# =========================
run_en_model <- function(target, remove_cols) {
  
  train.sub <- train.data[, !colnames(train.data) %in% remove_cols]
  test.sub  <- test.data[, !colnames(test.data) %in% remove_cols]
  
  formula <- as.formula(paste(target, "~ ."))
  
  set.seed(75)
  model <- train(
    formula,
    data = train.sub,
    method = "glmnet",
    trControl = control,
    preProcess = c("center", "scale"),
    tuneLength = 10   # auto-tune alpha & lambda
  )
  
  # Predictions
  pred_train <- predict(model, train.sub)
  pred_test  <- predict(model, test.sub)
  
  y_train <- train.sub[[target]]
  y_test  <- test.sub[[target]]
  
  range_y <- max(y_train) - min(y_train)
  
  nmae_train <- mean(abs(pred_train - y_train)) / range_y
  nmae_test  <- mean(abs(pred_test - y_test)) / range_y
  
  rsq_test <- 1 - sum((y_test - pred_test)^2) /
    sum((y_test - mean(y_test))^2)
  
  return(list(
    nmae_train = nmae_train,
    nmae_test = nmae_test,
    r2 = rsq_test,
    alpha = model$bestTune$alpha
  ))
}

# =========================
# 4. Run models
# =========================
res_rough <- run_en_model("roughness", c("tension_strenght", "elongation"))
res_tension <- run_en_model("tension_strenght", c("roughness", "elongation"))
res_elong <- run_en_model("elongation", c("roughness", "tension_strenght"))

# =========================
# 5. Create summary table
# =========================
results_table <- data.frame(
  Output_Parameter = c("Roughness", "Tensile Strength", "Elongation"),
  NMAE_Train = c(res_rough$nmae_train, res_tension$nmae_train, res_elong$nmae_train),
  NMAE_Test  = c(res_rough$nmae_test,  res_tension$nmae_test,  res_elong$nmae_test),
  R_squared         = c(res_rough$r2,         res_tension$r2,         res_elong$r2)
)

# Round values
results_table[, -1] <- round(results_table[, -1], 4)

print(results_table)