########################################  
##  Horizontal Stacked Bar Chart for BRICS Risk Groups - Relative Percentage Version
########################################  

# Load required packages  
packages <- c("dplyr", "ggplot2", "gridExtra", "tidyr", "grid")  
invisible(lapply(packages, function(x) {  
  if(!require(x, character.only = TRUE)) install.packages(x, quiet = TRUE)  
  library(x, character.only = TRUE)  
}))  

# Set BRICS cutoff value
brics_cutoff <- 2.4264

# Cohort list
cohorts <- c("RENJI_test", "North", "South", "PLCO", "TCGA")

# Define colors and labels - Non-BCR on the left, BCR≤1year on the right
color_mapping <- c(
  "Non-BCR" = "#191A70",           # Far left
  "BCR >3 years" = "#7F555F",      # 
  "BCR 1-3 years" = "#8B0025",     # 
  "BCR ≤1 year" = "#C91518"        # Far right
)

# Create stacked bar chart function
create_stacked_bar <- function(cohort_name, data_list) {
  
  # Get clinical data
  clinical_data <- data_list$clinical_data[[cohort_name]]
  
  if(is.null(clinical_data) || nrow(clinical_data) == 0) {
    cat("Warning: No clinical data available for", cohort_name, "\n")
    return(NULL)
  }
  
  # Get merged AI data
  ai_data <- data_list$prediction_data$merge[[cohort_name]]
  
  if(is.null(ai_data) || nrow(ai_data) == 0) {
    cat("Warning: No AI prediction data available for", cohort_name, "\n")
    return(NULL)
  }
  
  # Merge data
  combined_data <- ai_data %>%
    inner_join(clinical_data, by = "slide_id") %>%
    filter(!is.na(BCR), !is.na(`BCR Time(Months)`), !is.na(Clinical_AI_fusion_score)) %>%
    mutate(
      time_months = as.numeric(`BCR Time(Months)`),
      time_years = time_months / 12,
      event = as.numeric(BCR),
      original_score = Clinical_AI_fusion_score,
      ID = row_number()
    ) %>%
    filter(!is.na(time_years), time_years >= 0)
  
  # Check data volume
  if(nrow(combined_data) < 10) {
    cat("Warning: Insufficient data for", cohort_name, "(n=", nrow(combined_data), ")\n")
    return(NULL)
  }
  
  # Create BRICS risk groups
  plot_data <- combined_data %>%
    mutate(
      # BRICS risk groups
      brics_risk = ifelse(original_score >= brics_cutoff, "High risk", "Low risk"),
      brics_risk = factor(brics_risk, levels = c("Low risk", "High risk")),
      
      # BCR status groups - Non-BCR on the left, BCR≤1year on the right
      bcr_category = case_when(
        event == 0 ~ "Non-BCR",
        event == 1 & time_years <= 1 ~ "BCR ≤1 year",
        event == 1 & time_years > 1 & time_years <= 3 ~ "BCR 1-3 years",
        event == 1 & time_years > 3 ~ "BCR >3 years",
        TRUE ~ "Unknown"
      ),
      bcr_category = factor(bcr_category, levels = c("Non-BCR", "BCR >3 years", "BCR 1-3 years", "BCR ≤1 year"))
    ) %>%
    filter(bcr_category != "Unknown")
  
  # Calculate statistics
  summary_data <- plot_data %>%
    group_by(brics_risk, bcr_category) %>%
    summarise(count = n(), .groups = "drop") %>%
    group_by(brics_risk) %>%
    mutate(
      total = sum(count),
      percentage = count / total * 100
    ) %>%
    ungroup()
  
  # Print statistical information
  cat("\n=== ", cohort_name, " Statistics ===\n")
  cat("Total samples:", nrow(plot_data), "\n")
  cat("BRICS High risk:", sum(plot_data$brics_risk == "High risk"), 
      "(", round(sum(plot_data$brics_risk == "High risk")/nrow(plot_data)*100, 1), "%)\n")
  cat("BRICS Low risk:", sum(plot_data$brics_risk == "Low risk"), 
      "(", round(sum(plot_data$brics_risk == "Low risk")/nrow(plot_data)*100, 1), "%)\n")
  cat("BCR events:", sum(plot_data$event == 1), 
      "(", round(sum(plot_data$event == 1)/nrow(plot_data)*100, 1), "%)\n")
  cat("Non-BCR:", sum(plot_data$event == 0), 
      "(", round(sum(plot_data$event == 0)/nrow(plot_data)*100, 1), "%)\n")
  
  # Create horizontal stacked bar chart - using relative proportions
  p <- ggplot(summary_data, aes(x = brics_risk, y = percentage, fill = bcr_category)) +
    geom_bar(stat = "identity", position = "stack", color = "white", size = 0.3) +
    coord_flip() +  # Horizontal display
    scale_fill_manual(values = color_mapping, name = "BCR Status") +
    scale_y_continuous(limits = c(0, 101), breaks = seq(0, 100, 20), expand = c(0, 0)) +  # Set y-axis range to 0-100%
    labs(
      title = cohort_name,
      x = "BRICS Risk Group",
      y = "Percentage (%)"
    ) +
    theme_classic() +
    theme(
      text = element_text(size = 11, color = "black"),
      axis.text = element_text(size = 10, color = "black"),
      axis.title = element_text(size = 11, color = "black"),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position = "none",  # Remove individual plot legend
      axis.line = element_line(color = "black", size = 0.5),
      panel.grid.major.x = element_line(color = "gray90", size = 0.3),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 10, 10, 10)
    )
  
  # Add value labels - show percentage and sample count
  p <- p + geom_text(
    data = summary_data %>% filter(percentage > 3),  # Only show labels for >3% to avoid overlap
    aes(label = paste0(round(percentage, 1), "%\n(n=", count, ")")),
    position = position_stack(vjust = 0.5),
    size = 3,
    color = "white",
    fontface = "bold"
  )
  
  return(p)
}

# Create bar charts for all cohorts
plot_list <- list()
valid_cohorts <- c()

for(cohort in cohorts) {
  cat("Processing cohort:", cohort, "\n")
  
  plot <- create_stacked_bar(cohort, Fusion_regional_filtered)
  
  if(!is.null(plot)) {
    plot_list[[cohort]] <- plot
    valid_cohorts <- c(valid_cohorts, cohort)
  }
}

# Create shared legend
if(length(plot_list) > 0) {
  # Create a temporary plot to extract the legend
  temp_data <- data.frame(
    x = c("A", "A", "A", "A"),
    y = c(1, 2, 3, 4),
    fill = factor(c("Non-BCR", "BCR >3 years", "BCR 1-3 years", "BCR ≤1 year"), 
                  levels = c("Non-BCR", "BCR >3 years", "BCR 1-3 years", "BCR ≤1 year"))
  )
  
  temp_plot <- ggplot(temp_data, aes(x = x, y = y, fill = fill)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = color_mapping, name = "BCR Status") +
    theme_classic() +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      legend.key.size = unit(1, "cm")
    ) +
    guides(fill = guide_legend(title.position = "top", title.hjust = 0.5, nrow = 1))
  
  # Extract legend
  legend <- cowplot::get_legend(temp_plot)
  
  # Save to single-page PDF - single column arrangement, relative percentage version
  pdf("BRICS_Risk_BCR_Stacked_Bars_Percentage.pdf", width = 10, height = 14)
  
  # Create single-column grid layout
  grid.arrange(
    arrangeGrob(grobs = plot_list, ncol = 1, nrow = length(plot_list)),
    legend,
    ncol = 1,
    heights = c(0.92, 0.08)  # Main plot 92%, legend 8%
  )
  
  dev.off()
  
  cat("\n=== Single-column percentage stacked bar chart generation completed ===\n")
  cat("Output file: BRICS_Risk_BCR_Stacked_Bars_Percentage.pdf\n")
  cat("Number of cohorts plotted:", length(plot_list), "\n")
  cat("Layout: Single column (", length(plot_list), " rows x 1 column)\n")
  cat("Valid cohorts:", paste(valid_cohorts, collapse = ", "), "\n")
  cat("Chart type: Relative percentage (0-100%)\n")
  cat("BCR category order: Non-BCR (left) → BCR >3 years → BCR 1-3 years → BCR ≤1 year (right)\n")
  
} else {
  cat("Error: No plots generated\n")
}