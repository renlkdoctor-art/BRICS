########################################
##  Clinical + AI Fusion Model (Binary Risk Stratification - Strict p<0.05 Filtering)
########################################
setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/1. cox_model")

# Load required packages
pacman::p_load(survival, dplyr, car)

########################################
##  Parameter Configuration
########################################
load("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/1. data_preparation/4. cohort_resplit/regional_data_list.rdata")

DATA_LIST <- regional_data_list
MAGNIFICATION <- "merge"
TRAIN_COHORT <- "RENJI_train"
cohorts <- c("RENJI_train", "RENJI_test", "North", "South", "PLCO", "TCGA")

continuous_vars <- c("AI_score", "CAPRA_S_score")
categorical_vars <- c("psa_level", "pathologic_T_stage", "pathologic_N_stage", 
                      "ISUP", "residual_tumor_class", 
                      "p[Seminal Vesicle Invasion (SVI)]", "p[Extracapsular Extension (ECE)]")
excluded_vars <- c("CAPRA_S_score")  # Exclude from multivariate modeling
scale_features <- FALSE

########################################
##  Data Preprocessing
########################################
prep_data <- function(data_list, cohort, mag = "merge", scale = FALSE, 
                      ref_stats = NULL, excluded_vars = NULL) {
  
  clin <- data_list$clinical_data[[cohort]]
  if (is.null(clin) || nrow(clin) == 0) return(NULL)
  
  # Merge AI scores
  ai_data <- data_list$prediction_data[[mag]][[cohort]]
  if (!is.null(ai_data)) {
    clin <- left_join(clin, select(ai_data, slide_id, AI_score), by = "slide_id")
  }
  
  # Create PSA level categories
  if ("psa_value" %in% colnames(clin)) {
    clin$psa_level <- cut(as.numeric(clin$psa_value), 
                          breaks = c(-Inf, 6, 10, 20, Inf),
                          labels = c("0", "1", "2", "3"), right = FALSE)
    clin$psa_level <- as.numeric(as.character(clin$psa_level))
  }
  
  # Select variables (include all candidate variables)
  target_vars <- c(categorical_vars, continuous_vars, "BCR", "BCR Time(Months)")
  available_vars <- intersect(target_vars, colnames(clin))
  
  data <- clin %>%
    select(slide_id, all_of(available_vars)) %>%
    mutate(BCR_TimeMonths = as.numeric(`BCR Time(Months)`),
           BCR = as.numeric(BCR))
  
  # Standardize encoding
  if ("pathologic_T_stage" %in% colnames(data)) {
    data$pathologic_T_stage <- case_when(
      data$pathologic_T_stage == "pT2" ~ 0,
      data$pathologic_T_stage == "pT3a" ~ 1,
      data$pathologic_T_stage == "pT3b" ~ 2,
      data$pathologic_T_stage == "pT4" ~ 3,
      TRUE ~ as.numeric(data$pathologic_T_stage)
    )
  }
  
  if ("pathologic_N_stage" %in% colnames(data)) {
    data$pathologic_N_stage <- case_when(
      data$pathologic_N_stage == "pN0+pNx" ~ 0,
      data$pathologic_N_stage == "pN1" ~ 1,
      TRUE ~ as.numeric(data$pathologic_N_stage)
    )
  }
  
  # Convert other variables to numeric
  for (var in c(categorical_vars, continuous_vars)) {
    if (var %in% colnames(data)) {
      data[[var]] <- as.numeric(as.character(data[[var]]))
    }
  }
  
  # Standardize continuous variables
  stats <- list(means = list(), sds = list())
  if (scale) {
    scale_vars <- intersect(continuous_vars, colnames(data))
    for (var in scale_vars) {
      if (cohort == TRAIN_COHORT) {
        valid <- data[[var]][!is.na(data[[var]])]
        if (length(valid) > 1 && sd(valid) > 0) {
          stats$means[[var]] <- mean(valid)
          stats$sds[[var]] <- sd(valid)
          data[[var]] <- (data[[var]] - stats$means[[var]]) / stats$sds[[var]]
        }
      } else if (!is.null(ref_stats) && var %in% names(ref_stats$means)) {
        data[[var]] <- (data[[var]] - ref_stats$means[[var]]) / ref_stats$sds[[var]]
      }
    }
  }
  
  # Process survival data
  data$BCR_TimeMonths <- ifelse(data$BCR_TimeMonths == 0, 0.0001, data$BCR_TimeMonths)
  data <- filter(data, !is.na(BCR_TimeMonths), !is.na(BCR), BCR_TimeMonths > 0)
  
  cat(sprintf("%s: n=%d, events=%d\n", cohort, nrow(data), sum(data$BCR)))
  return(list(data = data, stats = stats))
}

########################################
##  Univariate Analysis
########################################
univariate_cox <- function(data, target_vars, p_threshold = 0.1) {
  surv_obj <- Surv(data$BCR_TimeMonths, data$BCR)
  predictors <- intersect(target_vars, colnames(data))
  
  cat("\n========== Univariate Cox Analysis ==========\n")
  
  # Safe variable name handling
  make_safe <- function(x) ifelse(grepl("[^A-Za-z0-9_.]", x), paste0("`", x, "`"), x)
  
  results <- lapply(predictors, function(var) {
    if (sum(!is.na(data[[var]])) < 10) return(NULL)
    
    tryCatch({
      safe_var <- make_safe(var)
      model <- coxph(as.formula(paste("surv_obj ~", safe_var)), data = data)
      summ <- summary(model)
      
      result <- data.frame(
        Variable = var,
        HR = summ$conf.int[1, "exp(coef)"],
        CI_lower = summ$conf.int[1, "lower .95"],
        CI_upper = summ$conf.int[1, "upper .95"],
        P_value = summ$coefficients[1, "Pr(>|z|)"]
      )
      
      cat(sprintf("  %-40s HR=%.2f (%.2f-%.2f), p=%s\n",
                  var, result$HR, result$CI_lower, result$CI_upper,
                  ifelse(result$P_value < 0.001, "<0.001", sprintf("%.3f", result$P_value))))
      return(result)
    }, error = function(e) {
      cat(sprintf("  %-40s Analysis failed: %s\n", var, e$message))
      return(NULL)
    })
  })
  
  results <- do.call(rbind, Filter(Negate(is.null), results))
  if (is.null(results)) stop("Univariate analysis failed")
  
  sig_vars <- results$Variable[results$P_value < p_threshold]
  cat("\nSignificant variables (p <", p_threshold, "):", paste(sig_vars, collapse = ", "), "\n")
  
  return(results)
}

########################################
##  Calculate C-index
########################################
calc_cindex <- function(time, event, risk) {
  valid <- !is.na(time) & !is.na(event) & !is.na(risk) & time > 0
  if (sum(valid) < 10) return(NA)
  
  surv_obj <- Surv(time[valid], event[valid])
  c1 <- tryCatch(concordance(surv_obj ~ risk[valid])$concordance, error = function(e) NA)
  c2 <- tryCatch(concordance(surv_obj ~ -risk[valid])$concordance, error = function(e) NA)
  
  return(ifelse(!is.na(c2) && c2 > c1, c2, c1))
}

########################################
##  Model Building (Strict p<0.05 Filtering Version)
########################################
build_cox_model <- function(train_data, all_vars, excluded_from_multivariate = NULL) {
  cat("\n========== Cox Model Building ==========\n")
  
  # 【Univariate Analysis】Include all candidate variables
  cat("\n【Univariate Analysis - All Variables】\n")
  uni_results <- univariate_cox(train_data, all_vars, p_threshold = 0.1)
  
  # 【Multivariate Modeling】Exclude specified variables
  multi_vars <- setdiff(all_vars, excluded_from_multivariate)
  cat("\n【Multivariate Modeling - Excluded Variables】:", 
      ifelse(length(excluded_from_multivariate) > 0, 
             paste(excluded_from_multivariate, collapse = ", "), 
             "None"), "\n")
  
  # Select variables for multivariate analysis from univariate results
  selected_vars <- uni_results$Variable[uni_results$P_value < 0.1 & 
                                          uni_results$Variable %in% multi_vars]
  
  # Ensure AI_score is included
  if ("AI_score" %in% multi_vars && !"AI_score" %in% selected_vars) {
    selected_vars <- c(selected_vars, "AI_score")
  }
  
  # Build initial model
  surv_obj <- Surv(train_data$BCR_TimeMonths, train_data$BCR)
  make_safe <- function(x) ifelse(grepl("[^A-Za-z0-9_.]", x), paste0("`", x, "`"), x)
  formula_str <- paste("surv_obj ~", paste(make_safe(selected_vars), collapse = " + "))
  
  cat("\nMultivariate model (initial):\n")
  standard_model <- coxph(as.formula(formula_str), data = train_data)
  print(summary(standard_model)$coefficients)
  
  # Backward stepwise optimization
  cat("\nBackward stepwise optimization...\n")
  step_model <- step(standard_model, direction = "backward", trace = 0)
  
  cat("\nModel after stepwise regression:\n")
  step_coef <- summary(step_model)$coefficients
  print(step_coef)
  
  # 【Critical】Second filtering: Remove variables with p >= 0.05
  cat("\n----------------------------------------\n")
  cat("【Strict Filtering: Remove variables with p≥0.05】\n")
  
  # Extract significant variables (p<0.05)
  sig_vars <- rownames(step_coef)[step_coef[, "Pr(>|z|)"] < 0.05]
  nonsig_vars <- rownames(step_coef)[step_coef[, "Pr(>|z|)"] >= 0.05]
  
  if (length(nonsig_vars) > 0) {
    cat("\nNon-significant variables detected, will be removed:\n")
    for (var in nonsig_vars) {
      p_val <- step_coef[var, "Pr(>|z|)"]
      hr <- step_coef[var, "exp(coef)"]
      cat(sprintf(" %s: HR=%.2f, p=%.3f\n", var, hr, p_val))
    }
    
    # Remove backticks for matching
    sig_vars_clean <- gsub("`", "", sig_vars)
    
    # Ensure AI_score is retained (core variable, even if p-value is slightly higher)
    if (!"AI_score" %in% sig_vars_clean && "AI_score" %in% selected_vars) {
      cat("\n   Force retain core variable: AI_score\n")
      sig_vars_clean <- c(sig_vars_clean, "AI_score")
    }
    
    # Refit final model
    formula_str_final <- paste("surv_obj ~", paste(make_safe(sig_vars_clean), collapse = " + "))
    final_model <- coxph(as.formula(formula_str_final), data = train_data)
    
    cat("\nFinal model (only significant variables p<0.05):\n")
    print(summary(final_model)$coefficients)
    
    cat("\nModel comparison:\n")
    cat(sprintf("  Stepwise model: AIC=%.1f, Variable count=%d\n", 
                AIC(step_model), nrow(step_coef)))
    cat(sprintf("  Final model:     AIC=%.1f, Variable count=%d\n", 
                AIC(final_model), length(sig_vars_clean)))
    cat(sprintf("  AIC difference: %.1f (difference<2 considered no substantial difference)\n", 
                AIC(final_model) - AIC(step_model)))
    
  } else {
    final_model <- step_model
    cat("\n All variables are significant (p<0.05), no further filtering needed\n")
    cat("\nAIC: Standard=", round(AIC(standard_model), 1), 
        ", Optimized=", round(AIC(step_model), 1), "\n")
  }
  
  cat("----------------------------------------\n")
  
  # Proportional hazards test
  ph_test <- tryCatch({
    test <- cox.zph(final_model)
    cat("\nProportional hazards test p=", round(test$table["GLOBAL", "p"], 4), "\n")
    test$table["GLOBAL", "p"]
  }, error = function(e) NA)
  
  return(list(
    model = final_model,
    coefficients = coef(final_model),
    univariate_results = uni_results,  # Contains univariate results for all variables
    ph_p = ph_test
  ))
}

########################################
##  Calculate Fusion Score
########################################
calculate_scores <- function(all_data, model_result) {
  cat("\n========== Calculate Fusion Score ==========\n")
  
  coefs <- model_result$coefficients
  
  # Remove backticks
  remove_backticks <- function(x) gsub("`", "", x)
  coefs_clean <- coefs
  names(coefs_clean) <- sapply(names(coefs), remove_backticks)
  
  lapply(names(all_data), function(cohort) {
    data <- all_data[[cohort]]
    data$Clinical_AI_fusion_score <- 0
    
    for (var in names(coefs_clean)) {
      if (var %in% colnames(data)) {
        data$Clinical_AI_fusion_score <- data$Clinical_AI_fusion_score + coefs_clean[var] * data[[var]]
      }
    }
    
    cat(sprintf("%s: range=[%.3f, %.3f]\n", cohort,
                min(data$Clinical_AI_fusion_score, na.rm = TRUE), 
                max(data$Clinical_AI_fusion_score, na.rm = TRUE)))
    
    select(data, slide_id, Clinical_AI_fusion_score)
  }) %>% setNames(names(all_data))
}

########################################
##  Determine Optimal Cutoff (on Training Set)
########################################
determine_optimal_cutoff <- function(train_data, target_rate = 0.10, time_point = 36) {
  
  cat("\n========== Determine Optimal Cutoff ==========\n")
  cat(sprintf("Target: Low-risk group %d-month recurrence rate ≈ %.1f%%\n", time_point, target_rate * 100))
  
  train_data <- train_data[order(train_data$Clinical_AI_fusion_score), ]
  candidate_cutoffs <- quantile(train_data$Clinical_AI_fusion_score, 
                                probs = seq(0.2, 0.8, by = 0.05), na.rm = TRUE)
  
  results <- lapply(candidate_cutoffs, function(cutoff) {
    low_risk <- train_data[train_data$Clinical_AI_fusion_score < cutoff, ]
    if (nrow(low_risk) < 20) return(NULL)
    
    surv_obj <- Surv(low_risk$BCR_TimeMonths, low_risk$BCR)
    fit <- survfit(surv_obj ~ 1)
    
    time_idx <- which.min(abs(fit$time - time_point))
    survival_prob <- if (length(time_idx) > 0) fit$surv[time_idx] else tail(fit$surv, 1)
    recurrence_rate <- 1 - survival_prob
    
    data.frame(
      cutoff = cutoff,
      n_low = nrow(low_risk),
      recurrence_rate = recurrence_rate,
      diff = abs(recurrence_rate - target_rate)
    )
  })
  
  results <- do.call(rbind, Filter(Negate(is.null), results))
  optimal_idx <- which.min(results$diff)
  optimal_cutoff <- results$cutoff[optimal_idx]
  
  cat(sprintf("\nOptimal cutoff: %.4f\n", optimal_cutoff))
  cat(sprintf("Low-risk group %d-month recurrence rate: %.1f%%\n", time_point, results$recurrence_rate[optimal_idx] * 100))
  cat(sprintf("Sample distribution: Low-risk=%d (%.1f%%), High-risk=%d (%.1f%%)\n",
              results$n_low[optimal_idx], 100 * results$n_low[optimal_idx] / nrow(train_data),
              nrow(train_data) - results$n_low[optimal_idx],
              100 * (1 - results$n_low[optimal_idx] / nrow(train_data))))
  
  return(optimal_cutoff)
}

########################################
##  Binary Risk Stratification (Apply Cutoff)
########################################
stratify_risk_binary <- function(all_data, scores, cutoff_value, time_point = 36) {
  
  cat("\n========== Binary Risk Stratification ==========\n")
  cat(sprintf("Cutoff: %.4f, Evaluation time: %d months\n\n", cutoff_value, time_point))
  
  results <- lapply(names(scores), function(cohort) {
    
    data <- inner_join(
      select(all_data[[cohort]], slide_id, BCR, BCR_TimeMonths),
      scores[[cohort]], by = "slide_id"
    )
    
    data$risk_group <- factor(
      ifelse(data$Clinical_AI_fusion_score < cutoff_value, "Low", "High"),
      levels = c("Low", "High")
    )
    
    # Calculate recurrence rate at time point
    time_rates <- data %>%
      group_by(risk_group) %>%
      summarise(
        N = n(),
        Events = sum(BCR),
        Rate_timepoint = {
          fit <- survfit(Surv(BCR_TimeMonths, BCR) ~ 1)
          time_idx <- which.min(abs(fit$time - time_point))
          surv_prob <- if (length(time_idx) > 0) fit$surv[time_idx] else tail(fit$surv, 1)
          round(100 * (1 - surv_prob), 1)
        },
        .groups = "drop"
      )
    
    # Calculate HR
    cox_model <- coxph(Surv(BCR_TimeMonths, BCR) ~ risk_group, data = data)
    summ <- summary(cox_model)
    hr <- summ$conf.int[1, "exp(coef)"]
    hr_ci <- summ$conf.int[1, c("lower .95", "upper .95")]
    p_val <- summ$coefficients[1, "Pr(>|z|)"]
    
    cat(sprintf("【%s】 n=%d\n", cohort, nrow(data)))
    print(time_rates)
    cat(sprintf("HR=%.2f (%.2f-%.2f), p=%s\n\n", hr, hr_ci[1], hr_ci[2],
                ifelse(p_val < 0.001, "<0.001", sprintf("%.3f", p_val))))
    
    list(data = data, stats = time_rates, hr = hr, hr_ci = hr_ci, p_value = p_val)
  })
  
  names(results) <- names(scores)
  return(results)
}

########################################
##  Main Workflow
########################################
cat("========================================\n")
cat("  Clinical + AI Fusion Model (Binary Stratification)\n")
cat("  Strict p<0.05 Filtering Version\n")
cat("========================================\n")

# 1. Data Preparation
cat("\n【Step 1】Data Preparation\n")
all_data <- list()
ref_stats <- NULL

for (cohort in cohorts) {
  result <- prep_data(DATA_LIST, cohort, MAGNIFICATION, scale_features, ref_stats)
  if (!is.null(result)) {
    all_data[[cohort]] <- result$data
    if (cohort == TRAIN_COHORT) ref_stats <- result$stats
  }
}

# 2. Model Building
cat("\n【Step 2】Model Building\n")

# All candidate variables (for univariate analysis)
all_candidate_vars <- c(continuous_vars, categorical_vars)

# Build model (univariate includes all variables, multivariate excludes CAPRA_S_score, strict p<0.05 filtering)
model_result <- build_cox_model(
  train_data = all_data[[TRAIN_COHORT]], 
  all_vars = all_candidate_vars,
  excluded_from_multivariate = excluded_vars
)

# 3. Calculate Scores
cat("\n【Step 3】Calculate Scores\n")
fusion_scores <- calculate_scores(all_data, model_result)

# 4. Determine cutoff and stratify
cat("\n【Step 4】Risk Stratification\n")
train_scores <- inner_join(
  select(all_data[[TRAIN_COHORT]], slide_id, BCR, BCR_TimeMonths),
  fusion_scores[[TRAIN_COHORT]], by = "slide_id"
)

optimal_cutoff <- determine_optimal_cutoff(train_scores, target_rate = 0.10, time_point = 36)
risk_groups <- stratify_risk_binary(all_data, fusion_scores, optimal_cutoff, time_point = 36)

# 5. Save Results
cat("\n【Step 5】Save Results\n")
save(model_result, file = "final_cox_model.rdata")
save(risk_groups, optimal_cutoff, file = "risk_stratification_binary.rdata")

# Update DATA_LIST
for (cohort in names(fusion_scores)) {
  if (is.null(DATA_LIST$prediction_data[[MAGNIFICATION]][[cohort]])) {
    DATA_LIST$prediction_data[[MAGNIFICATION]][[cohort]] <- fusion_scores[[cohort]]
  } else {
    DATA_LIST$prediction_data[[MAGNIFICATION]][[cohort]] <- 
      left_join(DATA_LIST$prediction_data[[MAGNIFICATION]][[cohort]],
                fusion_scores[[cohort]], by = "slide_id")
  }
}

Fusion_regional_filtered <- DATA_LIST
save(Fusion_regional_filtered, file = "Fusion_regional_filtered.rdata")

cat("\n========================================\n")
cat("          Model Building Complete!\n")
cat("========================================\n")
cat("Files saved:\n")
cat("  - final_cox_model.rdata (strict p<0.05 filtering)\n")
cat("  - risk_stratification_binary.rdata\n")
cat("  - Fusion_regional_filtered.rdata\n")
cat("========================================\n")