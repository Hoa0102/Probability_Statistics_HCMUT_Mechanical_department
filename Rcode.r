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
  geom_point(size = 3, alpha = 0.8) 
# 2. Bed Temperature vs Fan Speed
p_bed_fan <- ggplot(temp_data, aes(x = bed_temperature, y = fan_speed)) +
  geom_point(size = 3, color = "steelblue", alpha = 0.8) +
 # 3. Material vs Nozzle Temperature
p_mat_temp <- ggplot(temp_data, aes(x = factor(material, labels=c("ABS (0)", "PLA (1)")), y = nozzle_temperature, color = factor(material))) +
  geom_jitter(width = 0.15, size = 3, alpha = 0.8) +
# 4. Layer Height vs Roughness
p_rough <- ggplot(temp_data, aes(x = layer_height, y = roughness, color = factor(material))) +
  geom_point(size = 3, alpha = 0.8) 

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