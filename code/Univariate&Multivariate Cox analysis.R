########################################
##  Forest Plot and Nomogram for Fusion Model
########################################

# Load required packages
pacman::p_load(survival, forestplot, rms, ggplot2, ggsci, dplyr)

# Set working directory and load data
setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/1. cox_model")
load("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/1. cox_model/cox_model_environment_1011.RData")
setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/3. Forest_plots_nomogram")

# Get model and data
cox_model <- model_result$model
train_data <- all_data[[TRAIN_COHORT]]

########################################
##  Utility Functions
########################################

# Format variable names
format_var_name <- function(var) {
  case_when(
    grepl("AI_score", var) ~ "AI Score",
    grepl("CAPRA_S_score", var) ~ "CAPRA-S Score",
    grepl("psa_level", var) ~ "PSA Level",
    grepl("pathologic_T_stage", var) ~ "Pathologic T Stage",
    grepl("pathologic_N_stage", var) ~ "Pathologic N Stage", 
    grepl("ISUP", var) ~ "ISUP Grade",
    grepl("residual_tumor", var) ~ "Positive Surgical Margin",
    grepl("SVI|Seminal", var) ~ "Seminal Vesicle Invasion",
    grepl("ECE|Extracapsular", var) ~ "Extracapsular Extension",
    TRUE ~ var
  )
}

# Safe variable name handling (add backticks)
make_safe_formula <- function(var) {
  if (grepl("[^A-Za-z0-9_.]", var)) {
    return(paste0("`", var, "`"))
  }
  return(var)
}

# Remove backticks (for column name matching)
remove_backticks <- function(var) {
  gsub("`", "", var)
}

########################################
##  1. Univariate Cox Analysis
########################################
cat("\n========== Univariate Cox Analysis ==========\n")

# Define all candidate variables
candidate_vars <- c("AI_score", "CAPRA_S_score", "psa_level", 
                    "pathologic_T_stage", "pathologic_N_stage", 
                    "ISUP", "residual_tumor_class")

# Automatically find SVI and ECE columns
svi_cols <- grep("SVI|Seminal|vesicle", colnames(train_data), value = TRUE, ignore.case = TRUE)
ece_cols <- grep("ECE|Extracapsular|extension", colnames(train_data), value = TRUE, ignore.case = TRUE)

if (length(svi_cols) > 0) candidate_vars <- c(candidate_vars, svi_cols[1])
if (length(ece_cols) > 0) candidate_vars <- c(candidate_vars, ece_cols[1])

# Keep only existing variables
all_vars <- candidate_vars[candidate_vars %in% colnames(train_data)]

cat(sprintf("Number of actual candidate variables: %d\n", length(all_vars)))

# Survival object
surv_obj <- Surv(train_data$BCR_TimeMonths, train_data$BCR)

# Univariate analysis
uni_results_list <- list()

for (var in all_vars) {
  n_valid <- sum(!is.na(train_data[[var]]))
  if (n_valid < 10) next
  
  tryCatch({
    safe_var <- make_safe_formula(var)
    model <- coxph(as.formula(paste("surv_obj ~", safe_var)), data = train_data)
    summ <- summary(model)
    
    uni_results_list[[var]] <- data.frame(
      Variable = var,
      HR = summ$conf.int[1, "exp(coef)"],
      CI_lower = summ$conf.int[1, "lower .95"],
      CI_upper = summ$conf.int[1, "upper .95"],
      P_value = summ$coefficients[1, "Pr(>|z|)"],
      stringsAsFactors = FALSE
    )
    
    cat(sprintf("%-40s HR=%.2f (%.2f-%.2f), p=%s\n",
                var, uni_results_list[[var]]$HR, 
                uni_results_list[[var]]$CI_lower, 
                uni_results_list[[var]]$CI_upper,
                ifelse(uni_results_list[[var]]$P_value < 0.001, "<0.001", 
                       sprintf("%.3f", uni_results_list[[var]]$P_value))))
  }, error = function(e) {
    cat(sprintf("%-40s Analysis failed\n", var))
  })
}

uni_results <- do.call(rbind, uni_results_list)
cat(sprintf("\nSuccessfully analyzed %d variables\n", nrow(uni_results)))

########################################
##  2. Univariate Forest Plot
########################################
cat("\n========== Generate Univariate Forest Plot ==========\n")

uni_forest_data <- uni_results %>%
  mutate(Variable_clean = sapply(Variable, format_var_name)) %>%
  arrange(desc(HR))

uni_tabletext <- list(
  c("Variable", uni_forest_data$Variable_clean),
  c("HR (95% CI)", 
    paste0(sprintf("%.2f", uni_forest_data$HR), 
           " (", sprintf("%.2f", uni_forest_data$CI_lower), 
           "-", sprintf("%.2f", uni_forest_data$CI_upper), ")")),
  c("P value", 
    ifelse(uni_forest_data$P_value < 0.001, "<0.001", 
           sprintf("%.3f", uni_forest_data$P_value)))
)

pdf("Univariate_Cox_Forest_Plot.pdf", width = 12, height = max(6, nrow(uni_forest_data) * 0.6))
forestplot(uni_tabletext,
           mean = c(NA, uni_forest_data$HR),
           lower = c(NA, uni_forest_data$CI_lower),
           upper = c(NA, uni_forest_data$CI_upper),
           is.summary = c(TRUE, rep(FALSE, nrow(uni_forest_data))),
           boxsize = 0.35,
           lineheight = unit(0.8, "cm"),
           colgap = unit(3, "mm"),
           lwd.ci = 2,
           ci.vertices = TRUE,
           ci.vertices.height = 0.2,
           graphwidth = unit(3, "inches"),
           col = fpColors(box = "#ED0000", line = "#ED0000",
                          summary = "#00468B", hrz_lines = "#444444"),
           xlog = TRUE,
           xticks = c(0.5, 1, 2, 5, 10, 20, 40),
           grid = structure(c(1), gp = gpar(lty = 2, col = "#CCCCCC")),
           txt_gp = fpTxtGp(label = gpar(cex = 1.0, fontface = "bold"),    
                            ticks = gpar(cex = 0.9),    
                            xlab = gpar(cex = 1.0, fontface = "bold")),
           xlab = "Hazard Ratio (95% CI)",
           title = "Univariate Cox Regression Analysis for BCR")
dev.off()

cat(sprintf("Univariate forest plot saved (contains %d variables)\n", nrow(uni_forest_data)))

########################################
##  3. Multivariable Forest Plot
########################################
cat("\n========== Generate Multivariable Forest Plot ==========\n")

cox_summary <- summary(cox_model)
multi_forest_data <- data.frame(
  Variable = rownames(cox_summary$coefficients),
  HR = cox_summary$conf.int[, "exp(coef)"],
  CI_lower = cox_summary$conf.int[, "lower .95"],
  CI_upper = cox_summary$conf.int[, "upper .95"],
  P_value = cox_summary$coefficients[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
) %>%
  mutate(Variable = sapply(Variable, remove_backticks),  # Remove backticks
         Variable_clean = sapply(Variable, format_var_name)) %>%
  arrange(desc(HR))

multi_tabletext <- list(
  c("Variable", multi_forest_data$Variable_clean),
  c("HR (95% CI)", 
    paste0(sprintf("%.2f", multi_forest_data$HR), 
           " (", sprintf("%.2f", multi_forest_data$CI_lower), 
           "-", sprintf("%.2f", multi_forest_data$CI_upper), ")")),
  c("P value", 
    ifelse(multi_forest_data$P_value < 0.001, "<0.001", 
           sprintf("%.3f", multi_forest_data$P_value)))
)

pdf("Multivariable_Cox_Forest_Plot.pdf", width = 12, height = max(6, nrow(multi_forest_data) * 0.6))
forestplot(multi_tabletext,
           mean = c(NA, multi_forest_data$HR),
           lower = c(NA, multi_forest_data$CI_lower),
           upper = c(NA, multi_forest_data$CI_upper),
           is.summary = c(TRUE, rep(FALSE, nrow(multi_forest_data))),
           boxsize = 0.35,
           lineheight = unit(0.8, "cm"),
           colgap = unit(3, "mm"),
           lwd.ci = 2,
           ci.vertices = TRUE,
           ci.vertices.height = 0.2,
           graphwidth = unit(3, "inches"),
           col = fpColors(box = "#00468B", line = "#00468B",
                          summary = "#ED0000", hrz_lines = "#444444"),
           xlog = TRUE,
           xticks = c(0.5, 1, 2, 4, 8, 16),
           grid = structure(c(1), gp = gpar(lty = 2, col = "#CCCCCC")),
           txt_gp = fpTxtGp(label = gpar(cex = 1.0, fontface = "bold"),    
                            ticks = gpar(cex = 0.9),    
                            xlab = gpar(cex = 1.0, fontface = "bold")),
           xlab = "Hazard Ratio (95% CI)",
           title = "Multivariable Cox Regression: Clinical-AI Fusion Model for BCR")
dev.off()

cat(sprintf("Multivariable forest plot saved (contains %d variables)\n", nrow(multi_forest_data)))

########################################
##  4. Combined Forest Plot
########################################
cat("\n========== Generate Combined Forest Plot ==========\n")

combined_data <- uni_forest_data %>%
  left_join(multi_forest_data %>% 
              select(Variable, HR_multi = HR, CI_lower_multi = CI_lower, 
                     CI_upper_multi = CI_upper, P_multi = P_value),
            by = "Variable")

combined_tabletext <- list(
  c("Variable", combined_data$Variable_clean),
  c("Univariate HR (95% CI)", 
    paste0(sprintf("%.2f", combined_data$HR), 
           " (", sprintf("%.2f", combined_data$CI_lower), 
           "-", sprintf("%.2f", combined_data$CI_upper), ")")),
  c("P value", 
    ifelse(combined_data$P_value < 0.001, "<0.001", 
           sprintf("%.3f", combined_data$P_value))),
  c("Multivariable HR (95% CI)", 
    ifelse(!is.na(combined_data$HR_multi),
           paste0(sprintf("%.2f", combined_data$HR_multi), 
                  " (", sprintf("%.2f", combined_data$CI_lower_multi), 
                  "-", sprintf("%.2f", combined_data$CI_upper_multi), ")"),
           "Not included")),
  c("P value", 
    ifelse(!is.na(combined_data$P_multi),
           ifelse(combined_data$P_multi < 0.001, "<0.001", 
                  sprintf("%.3f", combined_data$P_multi)), ""))
)

pdf("Combined_Cox_Forest_Plot.pdf", width = 16, height = max(8, nrow(combined_data) * 0.7))
forestplot(combined_tabletext,
           mean = cbind(c(NA, combined_data$HR), 
                        c(NA, combined_data$HR_multi)),
           lower = cbind(c(NA, combined_data$CI_lower),
                         c(NA, combined_data$CI_lower_multi)),
           upper = cbind(c(NA, combined_data$CI_upper),
                         c(NA, combined_data$CI_upper_multi)),
           is.summary = c(TRUE, rep(FALSE, nrow(combined_data))),
           boxsize = 0.3,
           lineheight = unit(0.9, "cm"),
           colgap = unit(2, "mm"),
           lwd.ci = 2,
           ci.vertices = TRUE,
           ci.vertices.height = 0.15,
           graphwidth = unit(3.5, "inches"),
           col = fpColors(box = c("#ED0000", "#00468B"),
                          line = c("#ED0000", "#00468B"),
                          summary = "#444444", hrz_lines = "#444444"),
           xlog = TRUE,
           xticks = c(0.5, 1, 2, 5, 10, 20, 40),
           grid = structure(c(1), gp = gpar(lty = 2, col = "#CCCCCC")),
           txt_gp = fpTxtGp(label = gpar(cex = 0.95, fontface = "bold"),    
                            ticks = gpar(cex = 0.85),    
                            xlab = gpar(cex = 1.0, fontface = "bold")),
           xlab = "Hazard Ratio (95% CI)",
           title = "Cox Regression Analysis for BCR: Univariate and Multivariable Results",
           legend = c("Univariate", "Multivariable"),
           legend_args = fpLegend(pos = list(x = 0.85, y = 0.95), gp = gpar(cex = 0.9)))
dev.off()

cat(sprintf("Combined forest plot saved (contains %d variables)\n", nrow(combined_data)))

########################################
##  5. Nomogram - Fixed Version
########################################
cat("\n========== Generate Nomogram ==========\n")

# Get model variables and remove backticks
model_vars <- names(coef(cox_model))
model_vars_clean <- sapply(model_vars, remove_backticks)

cat("Model variables (after removing backticks):\n")
print(model_vars_clean)

# Check if all variables exist
cat("\nChecking variable existence:\n")
for (var in model_vars_clean) {
  exists <- var %in% colnames(train_data)
  cat(sprintf("  %s: %s\n", var, ifelse(exists, "Present", "Missing")))
}

# Prepare data (keep only model variables)
train_data_nomogram <- train_data[, c("BCR", "BCR_TimeMonths", model_vars_clean)]

# Set datadist and fit rms model
dd <- datadist(train_data_nomogram)
options(datadist = "dd")

# Build formula (safe handling for special characters)
safe_vars <- sapply(model_vars_clean, make_safe_formula)
rms_formula <- as.formula(paste("Surv(BCR_TimeMonths, BCR) ~", 
                                paste(safe_vars, collapse = " + ")))

cat("\nrms formula:\n")
print(rms_formula)

cat("\nFitting rms model...\n")
cph_model <- cph(rms_formula, data = train_data_nomogram, 
                 x = TRUE, y = TRUE, surv = TRUE)

# Verify coefficient consistency
cat("\nOriginal Cox model coefficients:\n")
print(round(coef(cox_model), 4))
cat("\nrms model coefficients:\n") 
print(round(coef(cph_model), 4))

# Create survival function
surv_func <- Survival(cph_model)

# Plot nomogram
pdf("Fusion_Model_Nomogram.pdf", width = 16, height = 10)
nom <- nomogram(cph_model,
                fun = list(
                  function(x) 1 - surv_func(12, x),   # 1-year
                  function(x) 1 - surv_func(24, x),   # 2-year
                  function(x) 1 - surv_func(36, x)    # 3-year
                ),
                fun.at = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
                funlabel = c("1-Year BCR Risk", "2-Year BCR Risk", "3-Year BCR Risk"),
                lp = TRUE, lp.at = seq(-2, 6, by = 1))

plot(nom, xfrac = 0.6, cex.axis = 0.8, cex.var = 1.0, lmgp = 0.2)
title("Clinical-AI Fusion Model Nomogram for BCR Prediction", cex.main = 1.2, font.main = 2)
dev.off()

cat("Nomogram saved\n")

########################################
##  Completion
########################################
cat("\n========================================\n")
cat("  Plot generation completed!\n")
cat("========================================\n")
cat("Generated files:\n")
cat("  1. Univariate_Cox_Forest_Plot.pdf\n")
cat("  2. Multivariable_Cox_Forest_Plot.pdf\n")
cat("  3. Combined_Cox_Forest_Plot.pdf\n")
cat("  4. Fusion_Model_Nomogram.pdf\n")
cat("========================================\n")