# =============================================================================
# Merge Data and Create Four Fusion Score Histograms - Nature Style (3-Year BCR Version)
# =============================================================================

library(ggplot2)
library(patchwork)  # For plot arrangement
library(dplyr)

# =============================================================================
# 1. Merge Data and Create 3-Year BCR Endpoint
# =============================================================================

cat("=== Starting Data Merging (3-Year BCR Version) ===\n")

# Merge test_cohort data and create 3-year BCR
test_merged <- Fusion_regional_filtered$prediction_data$merge$test_cohort %>%
  left_join(Fusion_regional_filtered$clinical_data$test_cohort, by = "slide_id") %>%
  filter(!is.na(Clinical_AI_fusion_score), !is.na(BCR), !is.na(`BCR Time(Months)`)) %>%
  mutate(
    bcr_time_months = as.numeric(`BCR Time(Months)`),
    # Create 3-year BCR endpoint
    bcr_3year = case_when(
      BCR == 1 & bcr_time_months <= 36 ~ 1,  # Recurrence within 3 years
      BCR == 0 | (BCR == 1 & bcr_time_months > 36) ~ 0,  # No recurrence or recurrence after 3 years
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(bcr_3year))

cat("Test cohort merging completed, 3-year BCR analysis sample size: ", nrow(test_merged), "\n")

# Merge RENJI_train data and create 3-year BCR
train_merged <- Fusion_regional_filtered$prediction_data$merge$RENJI_train %>%
  left_join(Fusion_regional_filtered$clinical_data$RENJI_train, by = "slide_id") %>%
  filter(!is.na(Clinical_AI_fusion_score), !is.na(BCR), !is.na(`BCR Time(Months)`)) %>%
  mutate(
    bcr_time_months = as.numeric(`BCR Time(Months)`),
    bcr_3year = case_when(
      BCR == 1 & bcr_time_months <= 36 ~ 1,
      BCR == 0 | (BCR == 1 & bcr_time_months > 36) ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  filter(!is.na(bcr_3year))

cat("RENJI train merging completed, 3-year BCR analysis sample size: ", nrow(train_merged), "\n")

# =============================================================================
# 2. Prepare Four Datasets (3-Year BCR Version)
# =============================================================================

# Data 1: Test Cohort all samples
test_data <- data.frame(
  score = test_merged$Clinical_AI_fusion_score,
  group = "Test Cohort",
  stringsAsFactors = FALSE
)

# Data 2: RENJI Train all samples
train_data <- data.frame(
  score = train_merged$Clinical_AI_fusion_score,
  group = "RENJI Train", 
  stringsAsFactors = FALSE
)

# Data 3: RENJI Train by 3-year BCR grouping
train_bcr_data <- train_merged %>%
  select(Clinical_AI_fusion_score, bcr_3year) %>%
  rename(score = Clinical_AI_fusion_score) %>%
  mutate(bcr_status = factor(ifelse(bcr_3year == 1, "3-Year BCR = 1", "3-Year BCR = 0"), 
                             levels = c("3-Year BCR = 0", "3-Year BCR = 1")))

# Data 4: Test Cohort by 3-year BCR grouping
test_bcr_data <- test_merged %>%
  select(Clinical_AI_fusion_score, bcr_3year) %>%
  rename(score = Clinical_AI_fusion_score) %>%
  mutate(bcr_status = factor(ifelse(bcr_3year == 1, "3-Year BCR = 1", "3-Year BCR = 0"),
                             levels = c("3-Year BCR = 0", "3-Year BCR = 1")))

# Output 3-year BCR data summary
cat("\n=== 3-Year BCR Data Summary ===\n")
cat("Test Cohort sample size: ", nrow(test_data), "\n")
cat("RENJI Train sample size: ", nrow(train_data), "\n")
cat("RENJI Train 3-year BCR groups: ", table(train_bcr_data$bcr_status), "\n")
cat("Test Cohort 3-year BCR groups: ", table(test_bcr_data$bcr_status), "\n")

# Output 3-year BCR rates
test_3year_bcr_rate <- mean(test_merged$bcr_3year)
train_3year_bcr_rate <- mean(train_merged$bcr_3year)
cat("Test Cohort 3-year BCR rate: ", round(test_3year_bcr_rate, 3), "\n")
cat("RENJI Train 3-year BCR rate: ", round(train_3year_bcr_rate, 3), "\n")

# =============================================================================
# 3. Nature Style Theme
# =============================================================================

nature_theme <- theme_classic() +
  theme(
    # Text
    text = element_text(size = 10, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 10, color = "black", face = "bold"),
    plot.title = element_text(size = 11, color = "black", face = "bold", hjust = 0.5),
    
    # Axes
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.4),
    
    # Background
    panel.background = element_blank(),
    plot.background = element_blank(),
    
    # Legend
    legend.position = "right",
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),
    
    # Margins
    plot.margin = margin(10, 10, 10, 10)
  )

# =============================================================================
# 4. Create Four Histograms (3-Year BCR Version)
# =============================================================================

# Histogram 1: RENJI Train overall
p1 <- ggplot(train_data, aes(x = score)) +
  geom_histogram(bins = 30, fill = "#762A83", alpha = 0.8, color = "white", size = 0.2) +
  labs(
    title = paste0("RENJI Train (n=", nrow(train_data), ")"),
    x = "Clinical-AI Fusion Score", 
    y = "Frequency"
  ) +
  nature_theme +
  scale_y_continuous(expand = c(0, 0, 0.05, 0)) +
  theme(legend.position = "none")

# Histogram 2: Test Cohort overall
p2 <- ggplot(test_data, aes(x = score)) +
  geom_histogram(bins = 30, fill = "#2166AC", alpha = 0.8, color = "white", size = 0.2) +
  labs(
    title = paste0("Test Cohort (n=", nrow(test_data), ")"),
    x = "Clinical-AI Fusion Score",
    y = "Frequency"
  ) +
  nature_theme +
  scale_y_continuous(expand = c(0, 0, 0.05, 0)) +
  theme(legend.position = "none")

# Histogram 3: RENJI Train by 3-year BCR
p3 <- ggplot(train_bcr_data, aes(x = score, fill = bcr_status)) +
  geom_histogram(bins = 25, alpha = 0.8, color = "white", size = 0.2, position = "identity") +
  scale_fill_manual(
    values = c("3-Year BCR = 0" = "#5AAE61", "3-Year BCR = 1" = "#D6604D"),
    name = "3-Year BCR"
  ) +
  labs(
    title = paste0("RENJI Train by 3-Year BCR (n=", nrow(train_bcr_data), ")"),
    x = "Clinical-AI Fusion Score",
    y = "Frequency"
  ) +
  nature_theme +
  scale_y_continuous(expand = c(0, 0, 0.05, 0))

# Histogram 4: Test Cohort by 3-year BCR
p4 <- ggplot(test_bcr_data, aes(x = score, fill = bcr_status)) +
  geom_histogram(bins = 25, alpha = 0.8, color = "white", size = 0.2, position = "identity") +
  scale_fill_manual(
    values = c("3-Year BCR = 0" = "#5AAE61", "3-Year BCR = 1" = "#D6604D"),
    name = "3-Year BCR"
  ) +
  labs(
    title = paste0("Test Cohort by 3-Year BCR (n=", nrow(test_bcr_data), ")"),
    x = "Clinical-AI Fusion Score",
    y = "Frequency"
  ) +
  nature_theme +
  scale_y_continuous(expand = c(0, 0, 0.05, 0))

# =============================================================================
# 5. Arrange Four Plots and Save
# =============================================================================

# Use patchwork for arrangement (2x2 layout)
combined_plot <- (p1 | p2) / (p3 | p4)

# Add main title
final_plot <- combined_plot + 
  plot_annotation(
    title = "Distribution of Clinical-AI Fusion Scores (3-Year BCR Analysis)",
    theme = theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5))
  )

# Save PDF
ggsave(
  filename = "Clinical_AI_Fusion_Score_Histograms_3Year_BCR.pdf",
  plot = final_plot,
  width = 12, 
  height = 8,
  dpi = 300
)

# =============================================================================
# 6. Output Detailed Statistics (3-Year BCR Version)
# =============================================================================

cat("\n=== Distribution Statistics (3-Year BCR Analysis) ===\n")

# Statistics function
get_stats <- function(data, name) {
  scores <- data$score
  cat(sprintf("%s:\n", name))
  cat(sprintf("  Sample size: %d\n", length(scores)))
  cat(sprintf("  Mean: %.3f\n", mean(scores, na.rm = TRUE)))
  cat(sprintf("  Median: %.3f\n", median(scores, na.rm = TRUE)))
  cat(sprintf("  Standard deviation: %.3f\n", sd(scores, na.rm = TRUE)))
  cat(sprintf("  Range: [%.3f, %.3f]\n\n", min(scores, na.rm = TRUE), max(scores, na.rm = TRUE)))
}

# 3-year BCR grouping statistics function
get_3year_bcr_stats <- function(data, cohort_name) {
  cat(sprintf("=== %s 3-Year BCR Group Statistics ===\n", cohort_name))
  
  bcr0_data <- data[data$bcr_status == "3-Year BCR = 0", ]
  bcr1_data <- data[data$bcr_status == "3-Year BCR = 1", ]
  
  cat(sprintf("3-Year BCR = 0: n=%d, mean=%.3f, SD=%.3f\n", 
              nrow(bcr0_data), mean(bcr0_data$score), sd(bcr0_data$score)))
  cat(sprintf("3-Year BCR = 1: n=%d, mean=%.3f, SD=%.3f\n", 
              nrow(bcr1_data), mean(bcr1_data$score), sd(bcr1_data$score)))
  
  if(nrow(bcr0_data) > 0 && nrow(bcr1_data) > 0) {
    t_test_result <- t.test(bcr1_data$score, bcr0_data$score)
    cat(sprintf("t-test p-value: %.6f (%s)\n", 
                t_test_result$p.value, 
                ifelse(t_test_result$p.value < 0.05, "Significant", "Not significant")))
  }
  cat("\n")
}

# Output statistics
get_stats(test_data, "Test Cohort overall")
get_stats(train_data, "RENJI Train overall")
get_3year_bcr_stats(train_bcr_data, "RENJI Train")
get_3year_bcr_stats(test_bcr_data, "Test Cohort")

cat("Plot saved as: Clinical_AI_Fusion_Score_Histograms_3Year_BCR.pdf\n")
cat("\n=== 3-Year BCR Distribution Analysis Completed ===\n")

# Display plot
print(final_plot)