# Load necessary packages
library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

# Simplified waterfall plot function - ggplot2 version (Lancet style)
create_waterfall_ggplot <- function(prediction_data, clinical_data, cohort_name, title_name, show_legend = FALSE) {
  
  # Get merged AI data
  ai_data <- prediction_data$merge[[cohort_name]]
  
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
  
  # BCR color assignment function
  assign_bcr_colors <- function(event, time_years) {
    case_when(
      # No recurrence/censored - dark blue
      event == 0 ~ "#191A70",
      # Recurrence - grouped by time
      event == 1 & time_years <= 1 ~ "#C91518",     # ≤1 year
      event == 1 & time_years > 1 & time_years <= 3 ~ "#8B0025",  # 1-3 years
      event == 1 & time_years > 3 ~ "#7F555F",      # >3 years
      TRUE ~ "#000000"  # Default black
    )
  }
  
  # Prepare AI Score data - sort by original score
  ai_plot_data <- combined_data %>%
    arrange(desc(original_score)) %>%
    mutate(
      ID_ordered = row_number(),
      BCR_color = assign_bcr_colors(event, time_years),
      BCR_status = case_when(
        event == 0 ~ "Non-BCR",
        event == 1 & time_years <= 1 ~ "BCR ≤1Y",
        event == 1 & time_years > 1 & time_years <= 3 ~ "BCR 1-3Y",
        event == 1 & time_years > 3 ~ "BCR >3Y",
        TRUE ~ "Unknown"
      ),
      # Set start and end points for each bar
      ymin = 2.4264,
      ymax = original_score
    )
  
  # Find X-axis position for 2.4264 threshold (boundary between high and low risk)
  threshold_position <- ai_plot_data %>%
    filter(original_score >= 2.4264) %>%
    nrow() + 0.5  # Between the last bar ≥2.4264 and the first bar <2.4264
  
  # Create ggplot (Lancet style)
  p <- ggplot(ai_plot_data, aes(x = ID_ordered, ymin = ymin, ymax = ymax, fill = BCR_status)) +
    geom_rect(aes(xmin = ID_ordered - 0.4, xmax = ID_ordered + 0.4), 
              color = NA) +
    geom_hline(yintercept = 2.4264, linetype = "dashed", color = "black", size = 0.5) +
    geom_vline(xintercept = threshold_position, linetype = "dashed", color = "black", size = 0.5) +
    scale_fill_manual(values = c(
      "Non-BCR" = "#191A70",
      "BCR ≤1Y" = "#C91518",
      "BCR 1-3Y" = "#8B0025",
      "BCR >3Y" = "#7F555F"
    )) +
    labs(
      title = title_name,
      y = "Clinical AI Fusion Score",
      fill = ""
    ) +
    theme_classic() +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position = if(show_legend) "top" else "none",
      legend.title = element_blank(),
      legend.text = element_text(size = 8),
      legend.key.size = unit(0.4, "cm"),
      legend.margin = margin(0, 0, 5, 0),
      axis.title.y = element_text(size = 10),
      axis.text.y = element_text(size = 8),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank(),
      axis.line.x = element_line(size = 0.3),
      axis.line.y = element_line(size = 0.3),
      panel.grid = element_blank(),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE))
  
  return(p)
}

# 1. Renji Test cohort (show legend)
plot1 <- create_waterfall_ggplot(
  prediction_data = Fusion_regional_filtered$prediction_data,
  clinical_data = Fusion_regional_filtered$clinical_data$RENJI_test,
  cohort_name = "RENJI_test",
  title_name = "Renji Test",
  show_legend = FALSE
)

# 2. North cohort (no legend)
plot2 <- create_waterfall_ggplot(
  prediction_data = Fusion_regional_filtered$prediction_data,
  clinical_data = Fusion_regional_filtered$clinical_data$North,
  cohort_name = "North",
  title_name = "North",
  show_legend = FALSE
)

# 3. South cohort (no legend)
plot3 <- create_waterfall_ggplot(
  prediction_data = Fusion_regional_filtered$prediction_data,
  clinical_data = Fusion_regional_filtered$clinical_data$South,
  cohort_name = "South",
  title_name = "South",
  show_legend = FALSE
)

# 4. PLCO cohort (no legend)
plot4 <- create_waterfall_ggplot(
  prediction_data = Fusion_regional_filtered$prediction_data,
  clinical_data = Fusion_regional_filtered$clinical_data$PLCO,
  cohort_name = "PLCO",
  title_name = "PLCO",
  show_legend = FALSE
)

# 5. TCGA cohort (no legend)
plot5 <- create_waterfall_ggplot(
  prediction_data = Fusion_regional_filtered$prediction_data,
  clinical_data = Fusion_regional_filtered$clinical_data$TCGA,
  cohort_name = "TCGA",
  title_name = "TCGA",
  show_legend = FALSE
)

# Arrange five plots by row
combined_plot <- grid.arrange(plot1, plot2, plot3, plot4, plot5, nrow = 5, 
                              top = textGrob("Clinical AI Fusion Score Waterfall Plots", 
                                             gp = gpar(fontsize = 14, fontface = "bold")))

# Save combined plot
ggsave("Combined_Waterfall_Plots_5cohorts.pdf", combined_plot, width = 4, height = 18, dpi = 300)

print(combined_plot)