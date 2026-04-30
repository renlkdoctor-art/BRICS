########################################  
##  KM Curve Generator - Clean Style Version (English)
########################################  

# Load required packages  
packages <- c("dplyr", "survival", "survminer", "ggplot2", "gridExtra", "grid")  
invisible(lapply(packages, function(x) {  
  if(!require(x, character.only = TRUE)) install.packages(x, quiet = TRUE)  
  library(x, character.only = TRUE)  
}))  

# Clean style theme definition
clean_theme <- theme_classic() + theme(  
  text = element_text(size = 10, color = "black"),   
  axis.text = element_text(size = 9, color = "black"),  
  axis.title = element_text(size = 11, color = "black"),
  axis.title.y = element_text(margin = margin(r = 15)),  # Increase right margin for Y-axis title
  plot.title = element_text(size = 12, face = "bold", hjust = 0.5),   
  axis.line = element_line(color = "black", size = 0.5),   
  panel.grid = element_blank()
)

# Risk table theme - increase margins
clean_risk_table_theme <- theme_void() + theme(
  axis.text.y = element_text(size = 8, color = "black", hjust = 1),
  plot.title = element_text(size = 10, face = "bold", hjust = 0)
)

########################################  
##  Parameter Settings  
########################################  

# Dataset specification  
DATA_LIST <- Fusion_regional_filtered  
MAGNIFICATION <- "merge"  

# Cohort selection  
cohorts <- c("RENJI_test", "North", "South", "PLCO", "TCGA")  

# Biomarker configuration  
roc_markers <- list(  
  Fusion_score = list(  
    name = "BRICS score",   
    source = "prediction",   
    column = "Clinical_AI_fusion_score",
    cutoff = 2.4264,
    type = "binary"  # Binary classification
  ),  
  
  capra_s = list(  
    name = "CAPRA-S",   
    source = "clinical",     
    column = "CAPRA_S_risk",  # Use risk classification column directly
    cutoff = NULL,  # No cutoff needed
    type = "tertiary"  # Tertiary classification
  )
)  

########################################  
##  Data Preprocessing Function  
########################################  

prep_data <- function(  
    data_list,   
    cohort,   
    mag = "merge",   
    markers = roc_markers,  
    time_column = "BCR Time(Months)",  
    event_column = "BCR",              
    time_unit = "month"                
) {  
  # Get clinical data  
  clin <- data_list$clinical_data[[cohort]]  
  if(is.null(clin) || nrow(clin) == 0) return(NULL)  
  
  # Check if key columns exist  
  if(!time_column %in% names(clin)) {  
    stop(paste("Time column", time_column, "not found in", cohort, "dataset"))  
  }  
  if(!event_column %in% names(clin)) {  
    stop(paste("Event column", event_column, "not found in", cohort, "dataset"))  
  }  
  
  # Dynamically create data transformation mapping  
  column_mapping <- list(  
    "time_years" = switch(time_unit,  
                          "month" = as.numeric(clin[[time_column]]) / 12,   
                          "year" = as.numeric(clin[[time_column]]),  
                          stop("Unsupported time unit")  
    ),  
    "event" = as.numeric(clin[[event_column]])  
  )  
  
  # Dynamically add biomarker columns  
  for(marker_key in names(markers)) {  
    marker_info <- markers[[marker_key]]  
    
    if(marker_info$source == "prediction") {  
      # Get data from prediction  
      pred_data <- data_list$prediction_data[[mag]][[cohort]]  
      
      if(!is.null(pred_data)) {  
        clin <- clin %>%   
          left_join(  
            pred_data %>% dplyr::select(slide_id, !!sym(marker_info$column)),   
            by = "slide_id"  
          )  
      }  
      
      if(marker_info$column %in% colnames(clin)) {
        column_mapping[[marker_key]] <- as.numeric(clin[[marker_info$column]])
      } else {
        warning(paste("Column not found:", marker_info$column))
      }
    } else {  
      # Get data from clinical  
      if(marker_info$column %in% colnames(clin)) {
        if(marker_info$type == "tertiary") {
          # For tertiary classification, convert to English
          capra_risk <- as.character(clin[[marker_info$column]])
          capra_risk_en <- case_when(
            capra_risk == "低风险" ~ "Low risk",
            capra_risk == "中风险" ~ "Intermediate risk", 
            capra_risk == "高风险" ~ "High risk",
            TRUE ~ capra_risk
          )
          column_mapping[[marker_key]] <- capra_risk_en
        } else {
          column_mapping[[marker_key]] <- as.numeric(clin[[marker_info$column]])
        }
      } else {
        warning(paste("Column not found:", marker_info$column))
      }
    }  
  }  
  
  # Data type conversion and cleaning  
  processed_data <- clin %>%  
    mutate(!!!column_mapping) %>%  
    filter(!is.na(time_years), !is.na(event), time_years >= 0) %>%  
    dplyr::select(time_years, event, all_of(names(markers)))  
  
  return(processed_data)  
}

########################################  
##  Global Cutoff Value Calculation  
########################################  

cat("=== Calculating global cutoff values based on training set ===\n")  

# Use RENJI_train cohort to calculate global cutoff values  
renji_train_data <- prep_data(DATA_LIST, "RENJI_train", MAGNIFICATION)  
if(is.null(renji_train_data)) {  
  stop("Error: RENJI_train data not available, cannot calculate global cutoff")  
}  

# Dynamically calculate median or use preset values for each biomarker  
global_cutoffs <- list()  
for(marker_key in names(roc_markers)) {  
  marker_info <- roc_markers[[marker_key]]  
  
  if(marker_info$type == "binary") {
    if(!is.null(marker_info$cutoff)) {  
      global_cutoffs[[marker_key]] <- marker_info$cutoff  
    } else {  
      global_cutoffs[[marker_key]] <- median(renji_train_data[[marker_key]], na.rm = TRUE)  
    }  
  } else {
    global_cutoffs[[marker_key]] <- NULL  # Tertiary classification doesn't need cutoff
  }
}  

cat("Training set statistics:\n")  
cat("  Sample size:", nrow(renji_train_data), "\n")  
cat("  Number of events:", sum(renji_train_data$event), "\n")  

# Check CAPRA-S risk distribution
if("capra_s" %in% names(renji_train_data)) {
  cat("CAPRA-S risk distribution:\n")
  print(table(renji_train_data$capra_s, useNA = "always"))
}

# Add BRICS score statistics
if("Fusion_score" %in% names(renji_train_data)) {
  brics_stats <- summary(renji_train_data$Fusion_score)
  cat("\nBRICS score statistics in training set:\n")
  cat("  Min:", format(brics_stats[1], digits=4), "\n")
  cat("  1st Quartile:", format(brics_stats[2], digits=4), "\n")
  cat("  Median:", format(brics_stats[3], digits=4), "\n")
  cat("  Mean:", format(mean(renji_train_data$Fusion_score, na.rm=TRUE), digits=4), "\n")
  cat("  3rd Quartile:", format(brics_stats[5], digits=4), "\n")
  cat("  Max:", format(brics_stats[6], digits=4), "\n")
  cat("  SD:", format(sd(renji_train_data$Fusion_score, na.rm=TRUE), digits=4), "\n\n")
}

########################################
##  KM Curve Generation Function - Clean Version
########################################

create_km_clean <- function(data, score_col, title_suffix, cohort, global_cutoff = NULL) {  
  # Filter valid data  
  valid_data <- data %>% filter(!is.na(.data[[score_col]]))  
  
  # Check sufficient sample size  
  if(nrow(valid_data) < 20) {  
    cat("  Warning:", cohort, title_suffix, "insufficient sample size (n=", nrow(valid_data), ")\n")  
    return(NULL)  
  }
  
  # Print BRICS score statistics (if Fusion_score)
  if(score_col == "Fusion_score") {
    brics_stats <- summary(valid_data$Fusion_score)
    cat("\nBRICS score statistics in", cohort, "cohort:\n")
    cat("  Sample size:", nrow(valid_data), "\n")
    cat("  Min:", format(brics_stats[1], digits=4), "\n")
    cat("  1st Quartile:", format(brics_stats[2], digits=4), "\n")
    cat("  Median:", format(brics_stats[3], digits=4), "\n")
    cat("  Mean:", format(mean(valid_data$Fusion_score, na.rm=TRUE), digits=4), "\n")
    cat("  3rd Quartile:", format(brics_stats[5], digits=4), "\n")
    cat("  Max:", format(brics_stats[6], digits=4), "\n")
    cat("  SD:", format(sd(valid_data$Fusion_score, na.rm=TRUE), digits=4), "\n")
    
    # Calculate statistics by BCR status
    cat("\n  BRICS score by BCR status in", cohort, "cohort:\n")
    for(event_status in c(0, 1)) {
      event_data <- valid_data[valid_data$event == event_status, ]
      if(nrow(event_data) > 0) {
        event_stats <- summary(event_data$Fusion_score)
        event_label <- ifelse(event_status == 1, "BCR", "No BCR")
        cat("    ", event_label, "(n=", nrow(event_data), "):\n")
        cat("      Median:", format(event_stats[3], digits=4), 
            ", Range:", format(event_stats[1], digits=4), "-", 
            format(event_stats[6], digits=4), "\n")
      }
    }
  }
  
  # Stratify based on biomarker type
  marker_info <- roc_markers[[score_col]]
  
  if(marker_info$type == "binary") {
    # Binary: based on cutoff
    valid_data$risk <- factor(  
      ifelse(valid_data[[score_col]] >= global_cutoff, "High risk", "Low risk"),   
      levels = c("Low risk", "High risk")  
    )
    colors <- c("Low risk" = "#3498DB", "High risk" = "#E74C3C")
    
    # Print BRICS score statistics for each risk group (if Fusion_score)
    if(score_col == "Fusion_score") {
      cat("\n  BRICS score by risk group in", cohort, "cohort (cutoff =", global_cutoff, "):\n")
      for(risk_group in c("Low risk", "High risk")) {
        risk_data <- valid_data[valid_data$risk == risk_group, ]
        if(nrow(risk_data) > 0) {
          risk_stats <- summary(risk_data$Fusion_score)
          cat("    ", risk_group, "(n=", nrow(risk_data), "):\n")
          cat("      Median:", format(risk_stats[3], digits=4), 
              ", Range:", format(risk_stats[1], digits=4), "-", 
              format(risk_stats[6], digits=4), "\n")
        }
      }
    }
  } else {
    # Tertiary: use existing classification directly
    valid_data$risk <- factor(valid_data[[score_col]], 
                              levels = c("Low risk", "Intermediate risk", "High risk"))
    colors <- c("Low risk" = "#3498DB", "Intermediate risk" = "#F39C12", "High risk" = "#E74C3C")
  }
  
  # Validate stratification results  
  risk_counts <- table(valid_data$risk)  
  cat("\n  ", cohort, title_suffix, "risk stratification:\n")
  print(risk_counts)
  
  # Check sample size in each group  
  if(any(risk_counts < 3)) {  
    cat("  Warning:", cohort, title_suffix, "insufficient sample size in some risk groups\n")  
    return(NULL)  
  }  
  
  # Fit survival model  
  fit <- survfit(Surv(time_years, event) ~ risk, data = valid_data)  
  
  # Calculate statistics
  if(marker_info$type == "binary") {
    # Binary: calculate HR and P-value
    cox_fit <- coxph(Surv(time_years, event) ~ risk, data = valid_data)  
    hr <- exp(coef(cox_fit))  
    hr_confint <- exp(confint(cox_fit))  
    hr_lower <- hr_confint[1, 1]  
    hr_upper <- hr_confint[1, 2]  
    
    # Calculate Cox regression P-value
    cox_summary <- summary(cox_fit)
    cox_pval <- cox_summary$coefficients[5]  # P-value in 5th column
    
    # Calculate Log-rank test P-value
    logrank_test <- survdiff(Surv(time_years, event) ~ risk, data = valid_data)
    logrank_pval <- 1 - pchisq(logrank_test$chisq, df = 1)
    
    # Format P-value
    format_pval <- function(p) {
      if(p < 0.0001) return("p<0.0001")
      else if(p < 0.001) return(paste0("p=", format(p, digits = 1, scientific = TRUE)))
      else return(paste0("p=", round(p, 4)))
    }
    
    # Create annotation text
    annotation_text <- paste0(
      "Low risk: reference\n",
      "High risk: HR ", round(hr, 2), " (95% CI ", round(hr_lower, 2), "–", round(hr_upper, 2), "), ", format_pval(cox_pval), "\n",
      "Log-rank test ", format_pval(logrank_pval)
    )
    
  } else {
    # Tertiary: calculate HR for each group relative to low risk
    cox_fit <- coxph(Surv(time_years, event) ~ risk, data = valid_data)  
    cox_summary <- summary(cox_fit)
    
    # Extract HR values and confidence intervals
    hr_values <- exp(coef(cox_fit))
    hr_confint <- exp(confint(cox_fit))
    
    # Extract P-values
    cox_pvals <- cox_summary$coefficients[, 5]  # P-values in 5th column
    
    # Calculate Log-rank test P-value
    logrank_test <- survdiff(Surv(time_years, event) ~ risk, data = valid_data)
    logrank_pval <- 1 - pchisq(logrank_test$chisq, df = 2)
    
    # Format P-value
    format_pval <- function(p) {
      if(p < 0.0001) return("p<0.0001")
      else if(p < 0.001) return(paste0("p=", format(p, digits = 1, scientific = TRUE)))
      else return(paste0("p=", round(p, 4)))
    }
    
    # Create annotation text
    annotation_lines <- c("Low risk: reference")
    
    # Intermediate risk vs Low risk
    if(length(hr_values) >= 1) {
      hr_int <- hr_values[1]  # Intermediate risk HR
      hr_int_lower <- hr_confint[1, 1]
      hr_int_upper <- hr_confint[1, 2]
      pval_int <- cox_pvals[1]
      
      annotation_lines <- c(annotation_lines,
                            paste0("Intermediate risk: HR ", round(hr_int, 2), " (95% CI ", 
                                   round(hr_int_lower, 2), "–", round(hr_int_upper, 2), "), ", 
                                   format_pval(pval_int)))
    }
    
    # High risk vs Low risk
    if(length(hr_values) >= 2) {
      hr_high <- hr_values[2]  # High risk HR
      hr_high_lower <- hr_confint[2, 1]
      hr_high_upper <- hr_confint[2, 2]
      pval_high <- cox_pvals[2]
      
      annotation_lines <- c(annotation_lines,
                            paste0("High risk: HR ", round(hr_high, 2), " (95% CI ", 
                                   round(hr_high_lower, 2), "–", round(hr_high_upper, 2), "), ", 
                                   format_pval(pval_high)))
    }
    
    # Add Log-rank test result
    annotation_lines <- c(annotation_lines, paste0("Log-rank test ", format_pval(logrank_pval)))
    
    annotation_text <- paste(annotation_lines, collapse = "\n")
  }
  
  # Create KM curve
  km_plot <- ggsurvplot(  
    fit,   
    data = valid_data,   
    pval = FALSE,  
    conf.int = TRUE,  
    risk.table = "abs_pct", # Add absolute and relative numbers in risk table              
    risk.table.height = 0.35,  # Increased from 0.3 to 0.35
    risk.table.y.text.col = FALSE,  
    fontsize= 2.5,  # Risk table font size
    risk.table.y.text = TRUE,  
    risk.table.title = "Number at risk",
    xlim = c(0, 5),
    break.time.by = 1,
    xlab = "Time (years)",   
    ylab = "BCR-free survival (%)",
    title = paste0(cohort, " - ", title_suffix),
    legend.title = "",
    legend.labs = names(colors),
    palette = colors,
    ggtheme = clean_theme,
    tables.theme = clean_risk_table_theme,
    size = 0.5,
    censor.size = 2,
    font.x = 10,        # X-axis font size
    font.y = 10,        # Y-axis font size
    font.tickslab = 9   # Tick label font size
  )
  
  # Modify plot and add annotation
  km_plot$plot <- km_plot$plot +  
    scale_y_continuous(labels = function(x) paste0(x*100)) +
    theme(
      legend.position = "none"
    ) +
    annotate("text", x = 0.2, y = 0.25,
             label = annotation_text,  
             size = 3, hjust = 0, vjust = 1, color = "black")
  
  return(km_plot)  
}

########################################
##  Generate All KM Curves
########################################

# Store KM curves  
km_plots_list <- list()  

# Iterate through each biomarker
for(marker_key in names(roc_markers)) {
  marker_info <- roc_markers[[marker_key]]
  
  # Determine cutoff value
  global_cutoff <- global_cutoffs[[marker_key]]
  
  cat("Biomarker:", marker_info$name, "\n")
  if(!is.null(global_cutoff)) {
    cat("  Using cutoff:", global_cutoff, "\n")
  } else {
    cat("  Using tertiary classification\n")
  }
  
  # Iterate through each cohort
  for(cohort in cohorts) {
    cat("Processing cohort:", cohort, "\n")
    
    # Get cohort data
    data <- prep_data(DATA_LIST, cohort, MAGNIFICATION)  
    
    if(is.null(data)) {  
      cat("  Warning:", cohort, "data not available\n")  
      next  
    }  
    
    # Create KM curve
    km_plot <- create_km_clean(data, marker_key, marker_info$name, cohort, global_cutoff)
    
    if(!is.null(km_plot)) {
      km_plots_list[[length(km_plots_list) + 1]] <- km_plot
    } else {
      cat("  Warning:", marker_info$name, "KM curve not generated for", cohort, "\n")
    }
  }
  
  cat("  Completed biomarker:", marker_info$name, "\n\n")
}

########################################
##  Save PDF File
########################################

if(length(km_plots_list) > 0) {
  num_features <- length(roc_markers)  # 2 biomarkers
  num_cohorts <- length(cohorts)       # 6 cohorts
  
  # Calculate page dimensions
  page_width <- 6 * num_features    
  page_height <- 5.3 * num_cohorts    
  
  # Save PDF
  pdf("KM_Curves_Clean_Style_English.pdf", width = page_width, height = page_height)  
  arrange_ggsurvplots(
    km_plots_list,
    ncol = num_features,   # 2 columns
    nrow = num_cohorts,    # 6 rows
    print = TRUE,
    risk.table.height = 0.16
  )  
  dev.off()
  
  # Summarize BRICS score statistics in a data frame and save as CSV
  if(any(names(roc_markers) == "Fusion_score")) {
    brics_stats_df <- data.frame(
      Cohort = character(),
      N = integer(),
      Median = numeric(),
      Min = numeric(),
      Max = numeric(),
      Mean = numeric(),
      SD = numeric(),
      stringsAsFactors = FALSE
    )
    
    for(cohort in cohorts) {
      data <- prep_data(DATA_LIST, cohort, MAGNIFICATION)
      if(!is.null(data) && "Fusion_score" %in% names(data)) {
        valid_data <- data %>% filter(!is.na(Fusion_score))
        if(nrow(valid_data) > 0) {
          brics_stats_df <- rbind(brics_stats_df, data.frame(
            Cohort = cohort,
            N = nrow(valid_data),
            Median = median(valid_data$Fusion_score, na.rm = TRUE),
            Min = min(valid_data$Fusion_score, na.rm = TRUE),
            Max = max(valid_data$Fusion_score, na.rm = TRUE),
            Mean = mean(valid_data$Fusion_score, na.rm = TRUE),
            SD = sd(valid_data$Fusion_score, na.rm = TRUE),
            stringsAsFactors = FALSE
          ))
        }
      }
    }
    
    # Add training set data
    data <- prep_data(DATA_LIST, "RENJI_train", MAGNIFICATION)
    if(!is.null(data) && "Fusion_score" %in% names(data)) {
      valid_data <- data %>% filter(!is.na(Fusion_score))
      if(nrow(valid_data) > 0) {
        brics_stats_df <- rbind(data.frame(
          Cohort = "RENJI_train",
          N = nrow(valid_data),
          Median = median(valid_data$Fusion_score, na.rm = TRUE),
          Min = min(valid_data$Fusion_score, na.rm = TRUE),
          Max = max(valid_data$Fusion_score, na.rm = TRUE),
          Mean = mean(valid_data$Fusion_score, na.rm = TRUE),
          SD = sd(valid_data$Fusion_score, na.rm = TRUE),
          stringsAsFactors = FALSE
        ), brics_stats_df)
      }
    }
    
    # Save statistics
    write.csv(brics_stats_df, "BRICS_Score_Statistics.csv", row.names = FALSE)
    cat("BRICS score statistics saved to BRICS_Score_Statistics.csv\n")
    
    # Print summary statistics table
    cat("\nBRICS Score Statistics Summary:\n")
    print(brics_stats_df)
  }
  
  cat("=== KM curve generation completed ===\n")
  cat("Output file: KM_Curves_Clean_Style_English.pdf\n")
  cat("Layout: ", num_cohorts, " rows x ", num_features, " columns\n")
  cat("BRICS score: Binary classification (cutoff = 2.4264)\n")
  cat("CAPRA-S: Tertiary classification (Low risk/Intermediate risk/High risk)\n")
  cat("Total number of plots: ", length(km_plots_list), "\n")
  
} else {
  cat("Error: No KM curves generated\n")
}