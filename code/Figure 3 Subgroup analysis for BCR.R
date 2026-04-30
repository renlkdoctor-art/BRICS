library(dplyr)
library(survival)
library(jstable)
library(forestploter)
library(tidyverse)
library(grid)
library(gridExtra)

setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/3. Model_evaluation/Figure_3/Subgroup_analysis_evaluation")
load("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/2. Full_cohort_generation/Fusion_regional_filtered_with_full_cohort.rdata")

clinical_data <- Fusion_regional_filtered$clinical_data$all_cohort %>%
  left_join(Fusion_regional_filtered$prediction_data$merge$all_cohort, by = "slide_id") %>%
  mutate(
    Risk_Group = case_when(
      Clinical_AI_fusion_score >= 2.4264 ~ "High Risk",
      Clinical_AI_fusion_score < 2.4264 ~ "Low Risk",
      TRUE ~ NA_character_
    ),
    Risk_Group = factor(Risk_Group, levels = c("Low Risk", "High Risk")),
    ISUP_group = case_when(ISUP >= 1 & ISUP <= 3 ~ "ISUP 1-3", ISUP >= 4 & ISUP <= 5 ~ "ISUP 4-5", TRUE ~ NA_character_),
    PSA_group = case_when(as.numeric(psa_value) > 10 ~ "PSA > 10", as.numeric(psa_value) <= 10 ~ "PSA ≤ 10", TRUE ~ NA_character_),
    T_stage_group = case_when(pathologic_T_stage == "pT2" ~ "T2", pathologic_T_stage %in% c("pT3a", "pT3b", "pT4") ~ "T3/T4", TRUE ~ NA_character_),
    N_stage_group = case_when(pathologic_N_stage == "pN0+pNx" ~ "pN0+pNx", pathologic_N_stage == "pN1" ~ "pN1", TRUE ~ NA_character_),
    PSM_group = case_when(residual_tumor_class == 1 ~ "Positive", residual_tumor_class == 0 ~ "Negative", TRUE ~ NA_character_),
    ECE_group = case_when(`p[Extracapsular Extension (ECE)]` == 1 ~ "Positive", `p[Extracapsular Extension (ECE)]` == 0 ~ "Negative", TRUE ~ NA_character_),
    SVI_group = case_when(`p[Seminal Vesicle Invasion (SVI)]` == 1 ~ "Positive", `p[Seminal Vesicle Invasion (SVI)]` == 0 ~ "Negative", TRUE ~ NA_character_),
    CAPRAS_group = case_when(
      CAPRA_S_risk == "Low risk" ~ "Low Risk",
      CAPRA_S_risk == "Intermediate risk" ~ "Intermediate Risk", 
      CAPRA_S_risk == "High risk" ~ "High Risk",
      TRUE ~ NA_character_
    )
  )

table(clinical_data$CAPRAS_group, clinical_data$Risk_Group)
table(clinical_data$PSA_group, clinical_data$Risk_Group)

# Helper function for calculating risk statistics
calc_risk_stat <- function(data, condition) {
  if (missing(condition)) {
    low_risk <- data %>% filter(Risk_Group == "Low Risk")
    high_risk <- data %>% filter(Risk_Group == "High Risk")
  } else {
    filtered_data <- data %>% filter(eval(parse(text = condition)))
    low_risk <- filtered_data %>% filter(Risk_Group == "Low Risk")
    high_risk <- filtered_data %>% filter(Risk_Group == "High Risk")
  }
  
  low_events <- sum(low_risk$BCR, na.rm = TRUE)
  low_total <- nrow(low_risk)
  high_events <- sum(high_risk$BCR, na.rm = TRUE)
  high_total <- nrow(high_risk)
  
  low_pct <- ifelse(low_total == 0, 0, low_events/low_total*100)
  high_pct <- ifelse(high_total == 0, 0, high_events/high_total*100)
  
  return(c(
    paste0(low_events, "/", low_total, " (", round(low_pct, 1), "%)"),
    paste0(high_events, "/", high_total, " (", round(high_pct, 1), "%)")
  ))
}

# Survival analysis
covariates <- c("ISUP_group", "PSA_group", "T_stage_group", "N_stage_group", "PSM_group", "ECE_group", "SVI_group", "CAPRAS_group")
clinical_data$`BCR Time(Months)` <- as.numeric(clinical_data$`BCR Time(Months)`)

res <- TableSubgroupMultiCox(formula = Surv(`BCR Time(Months)`, BCR) ~ Risk_Group,
                             data = clinical_data,
                             var_subgroups = covariates,
                             line = FALSE,
                             event = TRUE)

# Data preprocessing
res_processed <- res %>%
  mutate(
    across(everything(), ~ifelse(is.na(.), " ", .)),  
    `Hazard Ratio (95% CI)` = ifelse(`Point Estimate` == " ", " ",   
                                     paste0(`Point Estimate`, " (", Lower, "-", Upper, ")", sep = "")),  
    across(c(`Point Estimate`, Lower, Upper), ~as.numeric(.))  
  ) %>%
  rename("Subgroup" = "Variable") %>%
  mutate(
    BCR = `Count`, 
    Non_BCR = NA  
  )

# Change first row name to "All patients"
res_processed$Subgroup[1] <- "All patients"

# Create risk condition list based on actual Cox regression results (matching TableSubgroupMultiCox output order)
risk_conditions <- c(
  "TRUE",  # 1. All patients
  "TRUE",  # 2. ISUP_group - group header
  "ISUP_group == 'ISUP 1-3'",  # 3. ISUP 1-3
  "ISUP_group == 'ISUP 4-5'",  # 4. ISUP 4-5
  "TRUE",  # 5. PSA_group - group header
  "PSA_group == 'PSA > 10'",   # 6. PSA > 10
  "PSA_group == 'PSA ≤ 10'",   # 7. PSA ≤ 10
  "TRUE",  # 8. T_stage_group - group header
  "T_stage_group == 'T2'",     # 9. T2
  "T_stage_group == 'T3/T4'",  # 10. T3/T4
  "TRUE",  # 11. N_stage_group - group header
  "N_stage_group == 'pN0+pNx'", # 12. pN0+pNx
  "N_stage_group == 'pN1'",     # 13. pN1
  "TRUE",  # 14. PSM_group - group header
  "PSM_group == 'Negative'",    # 15. PSM Negative
  "PSM_group == 'Positive'",    # 16. PSM Positive
  "TRUE",  # 17. ECE_group - group header
  "ECE_group == 'Negative'",    # 18. ECE Negative
  "ECE_group == 'Positive'",    # 19. ECE Positive
  "TRUE",  # 20. SVI_group - group header
  "SVI_group == 'Negative'",    # 21. SVI Negative
  "SVI_group == 'Positive'",    # 22. SVI Positive
  "TRUE",  # 23. CAPRAS_group - group header
  "CAPRAS_group == 'High Risk'",         # 24. CAPRA-S High Risk (corrected: matching actual order)
  "CAPRAS_group == 'Intermediate Risk'", # 25. CAPRA-S Intermediate Risk
  "CAPRAS_group == 'Low Risk'"           # 26. CAPRA-S Low Risk (corrected: matching actual order)
)

# Calculate risk statistics for each row
n_rows <- nrow(res_processed)
Low_Risk_col <- c()
High_Risk_col <- c()

for (i in 1:min(length(risk_conditions), n_rows)) {
  condition <- risk_conditions[i]
  
  if (condition == "TRUE" && i > 1) {
    Low_Risk_col <- c(Low_Risk_col, " ")
    High_Risk_col <- c(High_Risk_col, " ")
  } else if (condition == "TRUE" && i == 1) {
    stats <- calc_risk_stat(clinical_data)
    Low_Risk_col <- c(Low_Risk_col, stats[1])
    High_Risk_col <- c(High_Risk_col, stats[2])
  } else {
    stats <- calc_risk_stat(clinical_data, condition)
    Low_Risk_col <- c(Low_Risk_col, stats[1])
    High_Risk_col <- c(High_Risk_col, stats[2])
  }
}

while (length(Low_Risk_col) < n_rows) {
  Low_Risk_col <- c(Low_Risk_col, " ")
  High_Risk_col <- c(High_Risk_col, " ")
}

# Add risk statistics columns
res_final <- res_processed %>%
  mutate(
    Low_Risk = Low_Risk_col[1:n_rows],
    High_Risk = High_Risk_col[1:n_rows]
  ) %>%
  rename("All patients" = "Count")

# Reorder: make all subgroups in "Low Risk → High Risk" order
# Define row pairs to swap (row numbers)
swap_pairs <- list(
  c(6, 7),   # PSA: swap PSA > 10 and PSA ≤ 10
  c(24, 26)  # CAPRAS: swap High Risk and Low Risk (keep Intermediate in the middle)
)

# Execute row swapping
for (pair in swap_pairs) {
  temp_row <- res_final[pair[1], ]
  res_final[pair[1], ] <- res_final[pair[2], ]
  res_final[pair[2], ] <- temp_row
}

print("Final result (reordered):")
print(res_final[, c("Subgroup", "All patients", "Low_Risk", "High_Risk")])

# Process forest_data
forest_data <- res_final %>%
  mutate(  
    `Point Estimate` = as.numeric(`Point Estimate`),  
    sizes = abs(`Point Estimate`),  
    sizes = ifelse(sizes == 0, 0.5, sizes),  
    `HR (95% CI)` = ifelse(is.na(`Point Estimate`), "", 
                           sprintf("%.2f (%.2f - %.2f)", `Point Estimate`, Lower, Upper)),
    ` ` = paste(rep(" ", 20), collapse = " ")
  )

# Select required columns
selected_columns <- c("Subgroup", "All patients", "Low_Risk", "High_Risk", " ", "HR (95% CI)", "P value")

# Create forest plot
p <- forest(  
  forest_data[, selected_columns],
  est = as.numeric(forest_data$`Point Estimate`),  
  lower = as.numeric(forest_data$Lower),
  upper = as.numeric(forest_data$Upper),
  sizes = 0.5,  
  ci_column = 5,
  ref_line = 1,  
  arrow_lab = c("Low Risk Better", "High Risk Better"),  
  xlim = c(0, 6), 
  footnote = "Subgroup analysis of Cox regression\nHazard Ratio with 95% Confidence Interval"  
)

# Insert main title for risk stratification
p <- insert_text(  
  p,  
  text = "Events/patients (n)",  
  row = 1,   
  col = c(3, 4),  
  just = "center",  
  gp = gpar(cex = 0.7, col = "black", fontface = "bold")  
)  

# Display plot
print(p)

# Save plot
ggsave(  
  filename = "forest_plot.pdf",   
  plot = p,  
  width = 16,   
  height = 12,
  dpi = 300  
)