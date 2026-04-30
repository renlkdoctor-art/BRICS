library(ggsankey)
library(tidyverse)
library(grid)
library(gridExtra)

setwd("D:/science_data/Urology/project/project8_BCR_recurrence/data/Article_final_code_1008/3. Model_evaluation/Figure_2")
cohorts <- c("RENJI_test", "North", "South", "PLCO", "TCGA")

# Generate BRICS × CAPRA-S contingency table
for(cohort in cohorts) {
  merged_data <- Fusion_regional_filtered$prediction_data$merge[[cohort]] %>%
    inner_join(Fusion_regional_filtered$clinical_data[[cohort]], by = "slide_id") %>%
    mutate(
      CAPRA_S_score = as.numeric(as.character(CAPRA_S_score)),  # Core fix
      BRICS_risk = ifelse(Clinical_AI_fusion_score >= 2.4264, "High risk", "Low risk"),
      CAPRA_S_detailed = case_when(
        CAPRA_S_score <= 2 ~ "Low risk",
        CAPRA_S_score == 3 ~ "Intermediate risk-1",
        CAPRA_S_score == 4 ~ "Intermediate risk-2", 
        CAPRA_S_score == 5 ~ "Intermediate risk-3",
        CAPRA_S_score >= 6 ~ "High risk"
      )
    )
  
  cat("\nCohort:", cohort, "\n")
  print(addmargins(table(merged_data$BRICS_risk, merged_data$CAPRA_S_detailed)))
  cat("\n")
}

# Create alluvial diagram
create_alluvial <- function(cohort_name) {
  merged_data <- Fusion_regional_filtered$prediction_data$merge[[cohort_name]] %>%
    inner_join(Fusion_regional_filtered$clinical_data[[cohort_name]], by = "slide_id") %>%
    filter(!is.na(Clinical_AI_fusion_score), !is.na(CAPRA_S_score)) %>%
    mutate(
      CAPRA_S_score = as.numeric(as.character(CAPRA_S_score)),  # Core fix
      BRICS_risk = ifelse(Clinical_AI_fusion_score >= 2.4264, "High risk", "Low risk"),
      CAPRA_S_detailed = case_when(
        CAPRA_S_score <= 2 ~ "Low risk",
        CAPRA_S_score == 3 ~ "Intermediate risk-1",
        CAPRA_S_score == 4 ~ "Intermediate risk-2", 
        CAPRA_S_score == 5 ~ "Intermediate risk-3",
        CAPRA_S_score >= 6 ~ "High risk"
      )
    ) %>%
    select(BRICS_risk, CAPRA_S_detailed) %>%
    make_long(BRICS_risk, CAPRA_S_detailed)
  
  ggplot(merged_data, aes(x = x, next_x = next_x, node = node, 
                          next_node = next_node, fill = factor(node))) +
    geom_alluvial(flow.alpha = 0.6, width = 0.2, curve_type = "sigmoid") +
    scale_fill_manual(values = c("Low risk" = "#3498DB", 
                                 "Intermediate risk-1" = "#F39C12",
                                 "Intermediate risk-2" = "#F39C12",
                                 "Intermediate risk-3" = "#F39C12",
                                 "High risk" = "#E74C3C")) +
    scale_x_discrete(labels = c("BRICS", "CAPRA-S")) +
    theme_void() +
    theme(legend.position = "none",
          plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          axis.text.x = element_text(size = 10, face = "bold")) +
    labs(title = cohort_name)
}

# Generate and save plots
plots <- lapply(cohorts, create_alluvial)
combined <- do.call(grid.arrange, c(plots, list(ncol = 1)))
ggsave("BRICS_CAPRAS_Alluvial.pdf", combined, width = 4, height = 20, dpi = 300)