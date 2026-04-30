########################################
##  Complete Version
########################################
packages <- c("dplyr", "timeROC", "survival", "purrr", "stringr", "tibble", "openxlsx")
needs <- setdiff(packages, rownames(installed.packages()))
if(length(needs)) install.packages(needs, quiet = TRUE)
invisible(lapply(packages, library, character.only = TRUE))

# Parameter Configuration
time_points <- c(1, 2, 3)
time_names <- paste0(time_points, "y")
magnifications <- c("X10", "X20", "X40", "merge")
cohorts <- c("RENJI_train", "RENJI_test", "PLCO", "TCGA", "North", "South")

# Score types - code will automatically detect location
score_types <- list(
  "AI_score" = "AI_score",
  "CAPRA_S" = "CAPRA_S_score",
  "Clinical_AI_fusion" = "Clinical_AI_fusion_score"
)

setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/3. Model_evaluation/AUC_C_index_evaluation")
load("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/2. Model_building/2. Full_cohort_generation/Fusion_regional_filtered(with_full_cohort).rdata")

USE_DATA_LIST <- Fusion_regional_filtered
DATA_LIST_NAME <- "regional_filtered_FUSION"

# Function to detect time column
detect_time_col <- function(df) {
  cand <- c("BCR_TIME_UPDATE", "BCR Time(Months)", "BCR_Time_Months", "BCR_time", "time_months")
  cand[cand %in% colnames(df)][1]
}

# Function to calculate C-index
calc_cindex <- function(time, event, risk) {
  surv_obj <- Surv(time, event)
  
  get_c <- function(risk_vec){
    if("concordance" %in% ls("package:survival")){
      fit <- concordance(surv_obj ~ risk_vec)
      est <- as.numeric(fit$concordance)
      se <- if(!is.null(fit$std.err)) as.numeric(fit$std.err) else sqrt(as.numeric(fit$var))
    } else {
      fit <- survConcordance(surv_obj ~ risk_vec)
      est <- fit$concordance
      se <- sqrt(fit$var)
    }
    list(c = est, se = se)
  }
  
  c1 <- get_c(risk)
  c2 <- get_c(-risk)
  
  if(c2$c > c1$c){
    est <- c2$c; se <- c2$se; rev <- TRUE
  } else {
    est <- c1$c; se <- c1$se; rev <- FALSE
  }
  lower <- est - 1.96*se
  upper <- est + 1.96*se
  list(cindex = est, lower = lower, upper = upper, reversed = rev)
}

# Function to calculate metrics
calc_metrics <- function(merged_df, cohort, mag, score_name, score_col) {
  time_var <- detect_time_col(merged_df)
  if (is.na(time_var)) {
    message("[Skip] ", score_name, "-", mag, "-", cohort, ": No follow-up time column found")
    return(NULL)
  }
  
  if (!score_col %in% colnames(merged_df)) {
    message("[Skip] ", score_name, "-", mag, "-", cohort, ": ", score_col, " column not found")
    return(NULL)
  }
  
  fused <- merged_df %>% 
    mutate(
      time_raw = as.numeric(as.character(.data[[time_var]])),
      event_raw = as.numeric(as.character(BCR)),
      risk_raw = as.numeric(as.character(.data[[score_col]]))
    ) %>%
    transmute(
      time = ifelse(time_raw == 0, 0.0001, time_raw / 12),  # Replace 0 with 0.0001
      event = event_raw,
      risk = risk_raw
    ) %>%
    filter(!is.na(event) & !is.na(risk)) 
  
  if (length(unique(fused$event)) < 2 || nrow(fused) < 20) {
    message("[Skip] ", score_name, "-", mag, "-", cohort, ": Insufficient events or samples")
    return(NULL)
  }
  
  tryCatch({
    # Calculate TimeROC
    troc <- timeROC(T = fused$time, delta = fused$event, marker = fused$risk,
                    cause = 1, times = time_points, iid = TRUE)
    
    ci_auc <- confint(troc, level = 0.95)
    
    # Save AUC results to data frame
    auc_tbl <- tibble(
      Data_Source = DATA_LIST_NAME,
      Score_Type = score_name,
      Magnification = mag,
      Cohort = cohort, 
      Timepoint = time_names,
      AUC = round(troc$AUC, 3),
      Lower_CI = round(ci_auc$CI_AUC[, "2.5%"] / 100, 3),
      Upper_CI = round(ci_auc$CI_AUC[, "97.5%"] / 100, 3)
    ) %>% mutate(AUC_CI = sprintf("%.3f (%.3f-%.3f)", AUC, Lower_CI, Upper_CI))
    
    # Calculate C-index
    c_res <- calc_cindex(fused$time, fused$event, fused$risk)
    c_tbl <- tibble(
      Data_Source = DATA_LIST_NAME,
      Score_Type = score_name,
      Magnification = mag,
      Cohort = cohort,
      C_index = round(c_res$cindex, 3),
      Lower_CI = round(c_res$lower, 3),
      Upper_CI = round(c_res$upper, 3),
      N = nrow(fused), 
      Events = sum(fused$event),
      Reversed = c_res$reversed
    ) %>% mutate(CI = sprintf("%.3f (%.3f-%.3f)", C_index, Lower_CI, Upper_CI))
    
    return(list(auc = auc_tbl, cidx = c_tbl))
    
  }, error = function(e) {
    message("[Error] ", score_name, "-", mag, "-", cohort, ": ", e$message)
    return(NULL)
  })
}

# Initialize valid count matrix
valid_counts <- tibble(Score_Type = character(), Magnification = character(), Cohort = character(), Valid_Count = integer())

# Smart main loop
cat("\n=== Starting Intelligent Calculation ===\n")

auc_list <- list()
cindex_list <- list()

for(mag in magnifications) {
  for(cohort in cohorts) {
    clin_df <- USE_DATA_LIST$clinical_data[[cohort]]
    if(is.null(clin_df)) {
      message("[Skip] ", cohort, ": Clinical data not found")
      next
    }
    if(!all(c("slide_id","BCR") %in% colnames(clin_df))) {
      message("[Skip] ", cohort, ": Missing required columns")
      next
    }
    
    for(score_name in names(score_types)) {
      score_col <- score_types[[score_name]]
      
      # Intelligent variable location detection
      pred_df <- USE_DATA_LIST$prediction_data[[mag]][[cohort]]
      in_pred <- !is.null(pred_df) && score_col %in% colnames(pred_df)
      in_clin <- score_col %in% colnames(clin_df)
      
      if(in_pred) {
        pred_use <- pred_df %>% dplyr::select(slide_id, !!sym(score_col))
        merged <- clin_df %>% inner_join(pred_use, by = "slide_id")
        source_type <- "prediction"
      } else if(in_clin) {
        merged <- clin_df
        source_type <- "clinical"
      } else {
        message("[Skip] ", score_name, "-", mag, "-", cohort, ": Variable does not exist")
        next
      }
      
      # Calculate valid count for each feature
      valid_count <- merged %>% 
        summarise(
          Valid_Count = sum(!is.na(.data[[score_col]]))  # Calculate valid count
        )
      
      valid_counts <- valid_counts %>% 
        add_row(Score_Type = score_name, Magnification = mag, Cohort = cohort, Valid_Count = valid_count$Valid_Count)
      
      res <- calc_metrics(merged, cohort, mag, score_name, score_col)
      if(!is.null(res)){
        auc_list[[length(auc_list)+1]] <- res$auc
        cindex_list[[length(cindex_list)+1]] <- res$cidx
        message("[OK] ", score_name, "-", mag, "-", cohort, "(", source_type, ")",
                ifelse(res$cidx$Reversed," (reversed)",""))
      }
    }
  }
}

# Output valid count matrix after all calculations
cat("\n=== Feature Valid Count Summary ===\n")
print(valid_counts)

# Continue with remaining code sections
auc_table <- bind_rows(auc_list)
cindex_table <- bind_rows(cindex_list)

########################################
##  Save Excel Results
########################################
auc_filename <- paste0("AUC_Results_", DATA_LIST_NAME, ".xlsx")
cindex_filename <- paste0("Cindex_Results_", DATA_LIST_NAME, ".xlsx")

# AUC Results
wb_auc <- createWorkbook()
for(mag in magnifications) {
  auc_mag <- auc_table %>% filter(grepl(mag, Magnification, ignore.case = TRUE))
  if(nrow(auc_mag) > 0) {
    addWorksheet(wb_auc, paste0("AUC_", mag))
    writeData(wb_auc, paste0("AUC_", mag), auc_mag)
    addStyle(wb_auc, paste0("AUC_", mag), 
             createStyle(textDecoration = "bold"), rows = 1, cols = 1:ncol(auc_mag))
  }
}

# C-index Results  
wb_cindex <- createWorkbook()
for(mag in magnifications) {
  cindex_mag <- cindex_table %>% filter(grepl(mag, Magnification, ignore.case = TRUE))
  if(nrow(cindex_mag) > 0) {
    addWorksheet(wb_cindex, paste0("Cindex_", mag))
    writeData(wb_cindex, paste0("Cindex_", mag), cindex_mag)
    addStyle(wb_cindex, paste0("Cindex_", mag), 
             createStyle(textDecoration = "bold"), rows = 1, cols = 1:ncol(cindex_mag))
  }
}

saveWorkbook(wb_auc, auc_filename, overwrite = TRUE)
saveWorkbook(wb_cindex, cindex_filename, overwrite = TRUE)

cat("\n[OK] Files saved:\n")
cat("  -", auc_filename, "\n")
cat("  -", cindex_filename, "\n")

########################################
##  Results Overview
########################################
cat("\n=== C-index Results Overview ===\n")
for(score in names(score_types)) {
  cat("\n", score, ":\n")
  score_results <- cindex_table %>% 
    filter(Score_Type == score) %>%
    dplyr::select(Magnification, Cohort, C_index, CI, N, Events)
  print(score_results)
}