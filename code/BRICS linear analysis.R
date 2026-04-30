# =============================================================================
# Correlation Analysis Between 3-Year BCR Risk and Clinical-AI Fusion Score
# =============================================================================

library(ggplot2)
library(patchwork)
library(dplyr)
library(rms)        # Calibration curve
library(pROC)       # ROC curve
library(mgcv)       # GAM smoothing

# =============================================================================
# 1. Prepare Analysis Data and Create 3-Year BCR Endpoint
# =============================================================================

cat("=== Preparing 3-Year BCR Analysis Data ===\n")

# Merge test_cohort data (main analysis)
test_analysis <- Fusion_regional_filtered$prediction_data$merge$test_cohort %>%
  left_join(Fusion_regional_filtered$clinical_data$test_cohort, by = "slide_id") %>%
  filter(!is.na(Clinical_AI_fusion_score), !is.na(BCR), !is.na(`BCR Time(Months)`)) %>%
  select(slide_id, case_id, BCR, `BCR Time(Months)`, Clinical_AI_fusion_score) %>%
  mutate(
    # Convert BCR time to numeric
    bcr_time_months = as.numeric(`BCR Time(Months)`),
    # Create 3-year BCR endpoint: BCR=1 and time<=36 months, or follow-up>36 months without recurrence as 0
    bcr_3year = case_when(
      BCR == 1 & bcr_time_months <= 36 ~ 1,  # Recurrence within 3 years
      BCR == 0 | (BCR == 1 & bcr_time_months > 36) ~ 0,  # No recurrence or recurrence after 3 years
      TRUE ~ NA_real_
    ),
    # Convert score to probability
    predicted_prob = plogis(Clinical_AI_fusion_score),
    actual_bcr = bcr_3year  # Use 3-year BCR as outcome
  ) %>%
  filter(!is.na(bcr_3year))  # Exclude patients with undetermined 3-year BCR status

# Merge train data (validation analysis)
train_analysis <- Fusion_regional_filtered$prediction_data$merge$RENJI_train %>%
  left_join(Fusion_regional_filtered$clinical_data$RENJI_train, by = "slide_id") %>%
  filter(!is.na(Clinical_AI_fusion_score), !is.na(BCR), !is.na(`BCR Time(Months)`)) %>%
  select(slide_id, case_id, BCR, `BCR Time(Months)`, Clinical_AI_fusion_score) %>%
  mutate(
    bcr_time_months = as.numeric(`BCR Time(Months)`),
    bcr_3year = case_when(
      BCR == 1 & bcr_time_months <= 36 ~ 1,
      BCR == 0 | (BCR == 1 & bcr_time_months > 36) ~ 0,
      TRUE ~ NA_real_
    ),
    predicted_prob = plogis(Clinical_AI_fusion_score),
    actual_bcr = bcr_3year
  ) %>%
  filter(!is.na(bcr_3year))

cat("Test cohort 3-year analysis samples: ", nrow(test_analysis), "\n")
cat("Train cohort 3-year analysis samples: ", nrow(train_analysis), "\n")
cat("Test cohort 3-year BCR rate: ", round(mean(test_analysis$actual_bcr), 3), "\n")
cat("Train cohort 3-year BCR rate: ", round(mean(train_analysis$actual_bcr), 3), "\n")

# Display BCR time distribution and 3-year cutoff analysis
cat("\n=== BCR Time Distribution Analysis ===\n")
cat("Test cohort median BCR time (months): ", median(test_analysis$bcr_time_months[test_analysis$BCR == 1], na.rm = TRUE), "\n")
cat("Train cohort median BCR time (months): ", median(train_analysis$bcr_time_months[train_analysis$BCR == 1], na.rm = TRUE), "\n")

# Summarize BCR distribution
test_bcr_summary <- test_analysis %>%
  summarise(
    total_bcr = sum(BCR),
    bcr_3year = sum(actual_bcr),
    bcr_after_3year = sum(BCR == 1 & bcr_time_months > 36, na.rm = TRUE),
    .groups = 'drop'
  )

train_bcr_summary <- train_analysis %>%
  summarise(
    total_bcr = sum(BCR),
    bcr_3year = sum(actual_bcr),
    bcr_after_3year = sum(BCR == 1 & bcr_time_months > 36, na.rm = TRUE),
    .groups = 'drop'
  )

cat("Test cohort: Total BCR ", test_bcr_summary$total_bcr, " cases, 3-year BCR ", test_bcr_summary$bcr_3year, " cases, BCR after 3 years ", test_bcr_summary$bcr_after_3year, " cases\n")
cat("Train cohort: Total BCR ", train_bcr_summary$total_bcr, " cases, 3-year BCR ", train_bcr_summary$bcr_3year, " cases, BCR after 3 years ", train_bcr_summary$bcr_after_3year, " cases\n")

# Display score conversion overview
cat("\n=== Score Conversion Overview ===\n")
cat("Test cohort original score range: ", 
    round(min(test_analysis$Clinical_AI_fusion_score), 3), " to ", 
    round(max(test_analysis$Clinical_AI_fusion_score), 3), "\n")
cat("Test cohort converted probability range: ", 
    round(min(test_analysis$predicted_prob), 3), " to ", 
    round(max(test_analysis$predicted_prob), 3), "\n")

# =============================================================================
# 2. Logistic Regression Analysis
# =============================================================================

cat("\n=== Logistic Regression Analysis (3-Year BCR) ===\n")

# Test cohort logistic regression (using original score)
test_glm <- glm(actual_bcr ~ Clinical_AI_fusion_score, data = test_analysis, family = binomial)
test_summary <- summary(test_glm)

# Train cohort logistic regression  
train_glm <- glm(actual_bcr ~ Clinical_AI_fusion_score, data = train_analysis, family = binomial)
train_summary <- summary(train_glm)

cat("Test Cohort Logistic Regression (3-Year BCR vs original score):\n")
cat(sprintf("  Coefficient: %.4f, p-value: %.6f\n", 
            test_summary$coefficients[2,1], test_summary$coefficients[2,4]))
cat(sprintf("  OR: %.4f (95%% CI: %.4f-%.4f)\n", 
            exp(test_summary$coefficients[2,1]),
            exp(test_summary$coefficients[2,1] - 1.96*test_summary$coefficients[2,2]),
            exp(test_summary$coefficients[2,1] + 1.96*test_summary$coefficients[2,2])))

cat("Train Cohort Logistic Regression (3-Year BCR vs original score):\n")
cat(sprintf("  Coefficient: %.4f, p-value: %.6f\n", 
            train_summary$coefficients[2,1], train_summary$coefficients[2,4]))
cat(sprintf("  OR: %.4f (95%% CI: %.4f-%.4f)\n", 
            exp(train_summary$coefficients[2,1]),
            exp(train_summary$coefficients[2,1] - 1.96*train_summary$coefficients[2,2]),
            exp(train_summary$coefficients[2,1] + 1.96*train_summary$coefficients[2,2])))

# =============================================================================
# 3. ROC Analysis
# =============================================================================

cat("\n=== ROC Analysis (3-Year BCR) ===\n")

# Calculate ROC (using original score to predict 3-year BCR)
test_roc <- roc(test_analysis$actual_bcr, test_analysis$Clinical_AI_fusion_score, quiet = TRUE)
train_roc <- roc(train_analysis$actual_bcr, train_analysis$Clinical_AI_fusion_score, quiet = TRUE)

cat("Test Cohort AUC (3-year BCR): ", round(auc(test_roc), 4), "\n")
cat("Train Cohort AUC (3-year BCR): ", round(auc(train_roc), 4), "\n")

# =============================================================================
# 4. Calibration Curve (Fixed rms Version)
# =============================================================================

cat("\n=== Calibration Curve Analysis (3-Year BCR) ===\n")

# Test cohort calibration curve
dd_test <- datadist(test_analysis)
options(datadist = "dd_test")

test_lrm <- lrm(actual_bcr ~ predicted_prob, data = test_analysis, x = TRUE, y = TRUE)
test_cal <- calibrate(test_lrm, method = "boot", B = 50)

cat("Test cohort 3-year BCR calibration curve completed\n")

# Train cohort calibration curve
dd_train <- datadist(train_analysis) 
options(datadist = "dd_train")

train_lrm <- lrm(actual_bcr ~ predicted_prob, data = train_analysis, x = TRUE, y = TRUE)
train_cal <- calibrate(train_lrm, method = "boot", B = 50)

cat("Train cohort 3-year BCR calibration curve completed\n")

# =============================================================================
# 5. Create Visualization Plots
# =============================================================================

# Nature theme
nature_theme <- theme_classic() +
  theme(
    text = element_text(size = 10, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 10, color = "black", face = "bold"),
    plot.title = element_text(size = 11, color = "black", face = "bold", hjust = 0.5),
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.4),
    panel.background = element_blank(),
    plot.background = element_blank(),
    legend.position = "right",
    plot.margin = margin(10, 10, 10, 10)
  )

# Figure 1: Test cohort - 3-year BCR rate vs Score
p1 <- ggplot(test_analysis, aes(x = Clinical_AI_fusion_score, y = actual_bcr)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), 
              color = "#2166AC", fill = "#2166AC", alpha = 0.3, size = 1.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              color = "black", linetype = "dashed", se = FALSE, size = 1) +
  labs(
    title = paste0("Test Cohort - 3-Year BCR (AUC = ", round(auc(test_roc), 3), ")"),
    x = "Clinical-AI Fusion Score",
    y = "3-Year BCR Rate"
  ) +
  nature_theme +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format())

# Figure 2: Train cohort - 3-year BCR rate vs Score
p2 <- ggplot(train_analysis, aes(x = Clinical_AI_fusion_score, y = actual_bcr)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), 
              color = "#762A83", fill = "#762A83", alpha = 0.3, size = 1.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"),
              color = "black", linetype = "dashed", se = FALSE, size = 1) +
  labs(
    title = paste0("Train Cohort - 3-Year BCR (AUC = ", round(auc(train_roc), 3), ")"),
    x = "Clinical-AI Fusion Score",
    y = "3-Year BCR Rate"
  ) +
  nature_theme +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format())

# Figure 3: Grouped 3-year BCR rate (original score grouping)
test_analysis$score_group <- cut(test_analysis$Clinical_AI_fusion_score, 
                                 breaks = quantile(test_analysis$Clinical_AI_fusion_score, 
                                                   probs = seq(0, 1, 0.2)),
                                 labels = c("Q1", "Q2", "Q3", "Q4", "Q5"),
                                 include.lowest = TRUE)

group_stats <- test_analysis %>%
  group_by(score_group) %>%
  summarise(
    n = n(),
    bcr_3year_rate = mean(actual_bcr),
    bcr_3year_se = sqrt(bcr_3year_rate * (1 - bcr_3year_rate) / n),
    mean_score = mean(Clinical_AI_fusion_score),
    .groups = 'drop'
  )

p3 <- ggplot(group_stats, aes(x = score_group, y = bcr_3year_rate)) +
  geom_col(fill = "#5AAE61", alpha = 0.8, color = "white", width = 0.7) +
  geom_errorbar(aes(ymin = bcr_3year_rate - 1.96*bcr_3year_se, 
                    ymax = bcr_3year_rate + 1.96*bcr_3year_se),
                width = 0.2, size = 0.5) +
  geom_text(aes(label = paste0("n=", n)), vjust = -0.5, size = 3) +
  geom_text(aes(label = sprintf("%.3f", mean_score)), vjust = 1.2, size = 2.5, color = "white") +
  labs(
    title = "3-Year BCR Rate by Score Quintiles",
    x = "Score Quintiles (Test Cohort)",
    y = "3-Year BCR Rate"
  ) +
  nature_theme +
  scale_y_continuous(limits = c(0, max(group_stats$bcr_3year_rate + 1.96*group_stats$bcr_3year_se) * 1.1), 
                     labels = scales::percent_format())

# Figure 4: Calibration curve (3-year BCR)
create_calibration_data <- function(actual, predicted, n_bins = 10) {
  bins <- cut(predicted, 
              breaks = quantile(predicted, seq(0, 1, length.out = n_bins + 1)),
              include.lowest = TRUE)
  
  cal_data <- data.frame(
    predicted = predicted,
    actual = actual,
    bin = bins
  ) %>%
    group_by(bin) %>%
    summarise(
      mean_predicted = mean(predicted),
      mean_actual = mean(actual),
      se_actual = sqrt(mean_actual * (1 - mean_actual) / n()),
      n = n(),
      .groups = 'drop'
    ) %>%
    filter(n >= 5)
  
  return(cal_data)
}

test_cal_data <- create_calibration_data(test_analysis$actual_bcr, test_analysis$predicted_prob)
train_cal_data <- create_calibration_data(train_analysis$actual_bcr, train_analysis$predicted_prob)

cal_plot_data <- bind_rows(
  test_cal_data %>% mutate(cohort = "Test"),
  train_cal_data %>% mutate(cohort = "Train")
)

p4 <- ggplot(cal_plot_data, aes(x = mean_predicted, y = mean_actual, color = cohort)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_errorbar(aes(ymin = mean_actual - 1.96*se_actual, 
                    ymax = mean_actual + 1.96*se_actual),
                width = 0.02, alpha = 0.7) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2, size = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", size = 1) +
  scale_color_manual(values = c("Test" = "#2166AC", "Train" = "#762A83")) +
  labs(
    title = "Calibration Plot (3-Year BCR)",
    x = "Mean Predicted 3-Year BCR Risk",
    y = "Observed 3-Year BCR Rate",
    color = "Cohort"
  ) +
  nature_theme +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1))

# =============================================================================
# 7. Correlation Tests
# =============================================================================

cat("\n=== Correlation Analysis (3-Year BCR) ===\n")

# Pearson correlation
test_cor <- cor.test(test_analysis$Clinical_AI_fusion_score, test_analysis$actual_bcr, method = "pearson")
train_cor <- cor.test(train_analysis$Clinical_AI_fusion_score, train_analysis$actual_bcr, method = "pearson")

cat("Test Cohort Pearson correlation (3-year BCR):\n")
cat(sprintf("  r = %.4f, p-value = %.6f\n", test_cor$estimate, test_cor$p.value))

cat("Train Cohort Pearson correlation (3-year BCR):\n")
cat(sprintf("  r = %.4f, p-value = %.6f\n", train_cor$estimate, train_cor$p.value))

# =============================================================================
# 8. Arrange Plots and Save
# =============================================================================

# Create combined plot (2x2 layout)
combined_plot <- (p1 | p2) / (p3 | p4)

final_plot <- combined_plot + 
  plot_annotation(
    title = "3-Year BCR Risk vs Clinical-AI Fusion Score Correlation Analysis",
    theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
  )

# Save
ggsave(
  filename = "BCR_3Year_Score_Correlation_Analysis.pdf",
  plot = final_plot,
  width = 12,
  height = 10,
  dpi = 300
)

# Output group statistics
cat("\n=== Test Cohort 3-Year BCR Group Statistics ===\n")
print(group_stats)

cat("\n=== Analysis Completed ===\n")
cat("Plot saved as: BCR_3Year_Score_Correlation_Analysis.pdf\n")
cat("Includes the following analyses:\n")
cat("1. 3-year BCR rate vs fusion score GAM/logistic regression fit\n")
cat("2. Grouped 3-year BCR rate analysis\n") 
cat("3. 3-year BCR prediction calibration curve\n")
cat("4. 3-year BCR correlation test results\n")

# Display plot
print(final_plot)