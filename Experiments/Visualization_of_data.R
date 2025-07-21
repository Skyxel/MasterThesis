################################################################################
### Preconditioner Comparison (Vecchia) x3
################################################################################

#######
## Packages
#######

# Load the required libraries
library(ggplot2)
library(grid)
library(dplyr)

# rho
rho_values <- c(0.25 ,0.05, 0.01)

SDxTime <- TRUE
variance <- FALSE

precond_map <- list(
  "Sigma_inv_plus_BtWB"                  = c("VADU",            "P[VADU]"),
  "piv_chol_on_Sigma"                    = c("Pivoted Cholesky","P[LRAC]"),
  "predictive_process_plus_diagonal_200" = c("FITC",            "P[FITC_200]"),
  "incomplete_cholesky"                  = c("ZIRC",            "P[ZIRC]"),
  "incomplete_cholesky_SN_B"             = c("ZIRC SN B",       "P[ZIRC_SN_B]"),
  "incomplete_cholesky_SN_A"             = c("ZIRC SN Sigma",   "P[ZIRC_SN_A]"),
  "incomplete_cholesky_T"                = c("ZIRC T",          "P[ZIRC_T]"),
  "incomplete_cholesky_JM"               = c("ZIRC JM",         "P[ZIRC_JM]"),
  "incomplete_cholesky_TJM_01"           = c("ZIRC T + JM 0.1", "P[ZIRC_TJM_01]"),
  "incomplete_cholesky_TJM_001"          = c("ZIRC T + JM 0.01",     "P[ZIRC_TJM_001]"),
  "incomplete_cholesky_TJM_0001"         = c("ZIRC T + JM 0.001",     "P[ZIRC_TJM_0001]"),
  "hlfpc"                                = c("HLF PC",          "P[HLFPC]"),
  "hlfpc_nystroem_last"                  = c("HLF PC N Last",   "P[HLFPCNL]"),
  "hlfpc_nystroem_random"                = c("HLF PC N Random", "P[HLFPCNR]"),
  "hlfpc_pivoted_cholesky"               = c("HLF PC PC Fact",  "P[HLFPCPC]"),
  "hlfpc_lanczos"                        = c("HLF PC Lanczos",  "P[HLFPCL]"),
  "hlfpc_rlra"                           = c("HLF PC RLRA",     "P[HLFPCRLRA]")
)

# Preconditioners to test
PRECONDITIONER <- c("Sigma_inv_plus_BtWB"
                    , "piv_chol_on_Sigma"
                    , "predictive_process_plus_diagonal_200"
                    #,"incomplete_cholesky"
                    #,"incomplete_cholesky_SN_A"
                    #,"incomplete_cholesky_SN_B"
                    #,"incomplete_cholesky_T"
                    #,"incomplete_cholesky_JM"
                    #,"incomplete_cholesky_TJM_01"
                    #,"incomplete_cholesky_TJM_001"
                    #,"incomplete_cholesky_TJM_0001"
                    #,"incomplete_cholesky"
                    ,"hlfpc_rlra"
                    ,"hlfpc"
                    ,"hlfpc_nystroem_last"
                    #,"hlfpc_nystroem_random"
                    #,"hlfpc_pivoted_cholesky"
                    #,"hlfpc_lanczos"
)

AllResults <- data.frame()
AllResults_saved <- readRDS("AllResults.rds")

VL_times <- numeric(0)
VL_results <- numeric(0)

for (rho_true in rho_values){
  loaded_itresults <- AllResults_saved %>% 
    filter(rho == rho_true)
  
  VLresult <- loaded_itresults$negLL_VL[1]
  VLtime <- loaded_itresults$time_VL[1]
  
  VL_times <- c(VL_times, VLtime)
  VL_results <- c(VL_results, VLresult)
  
  Itresults <- data.frame()
  for (p in seq_along(PRECONDITIONER)) {
      Itresults <- loaded_itresults %>%
        filter(preconditioner == PRECONDITIONER[p])
      AllResults <- rbind(AllResults, Itresults)
  }
}

################################################################################
# Create a data frame containing VL_times for each rho

preconditioner_labels <- c()
preconditioners_levels <- c()

for (key in intersect(PRECONDITIONER, names(precond_map))) {
  preconditioner_labels <- c(preconditioner_labels, precond_map[[key]][1])
  preconditioners_levels <- c(preconditioners_levels, precond_map[[key]][2])
  AllResults$preconditioner[AllResults$preconditioner == key] <- precond_map[[key]][2]
}

AllResults$preconditioner <- factor(
  AllResults$preconditioner, 
  levels = preconditioners_levels,
  ordered = TRUE
)

AllResults$t <- as.factor(AllResults$t)

if(SDxTime){
  # Ensure `t` is numeric, not factor
  AllResults$t <- as.numeric(as.character(AllResults$t))
  
  agg_stats <- data.frame()
  # Group by rho, t, and preconditioner
  agg_stats <- AllResults %>%
    group_by(rho, t, preconditioner) %>%
    summarise(
      sd_negLL_diff = sd(negLL_diff, na.rm = TRUE),
      mean_time = mean(time, na.rm = TRUE),
      .groups = "drop"
    )
  
  agg_stats <- agg_stats %>%
    arrange(rho, preconditioner, t)
  
  VL_times_df <- data.frame(rho = unique(AllResults$rho), VL_time = VL_times)
  
  VL_times_ann <- VL_times_df %>%
    mutate(
      x = max(AllResults$time) -20,
      y = max(agg_stats$sd_negLL_diff) -2 ,
      label = paste0("Cholesky: ", signif(VL_time, 3), " s") 
    )
  
  # Plot
  p_time_sd <- ggplot(agg_stats, aes(x = mean_time, y = sd_negLL_diff, color = preconditioner, group = preconditioner)) +
    geom_path(linewidth = 1) +
    geom_point(size = 2) +
    geom_text(aes(label = t), vjust = -0.5, hjust = -0.2, size = 3, show.legend = FALSE) +
    facet_wrap(~rho, nrow = 1, scales = "fixed",
               labeller = label_bquote(rho == .(rho))) +
    scale_color_brewer(type = "qual", palette = 6, labels = preconditioner_labels) +
    scale_y_log10() +
    scale_x_log10() +
    labs(x = "Time (s)", y = "SD of difference to Cholesky") +
    theme_bw() +
    theme(
      text = element_text(size = 15),
      legend.position = "top",
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.title.x = element_text(margin = margin(t = 10)),
    )
  
  p_time_sd <- p_time_sd +
    geom_text(data = VL_times_ann, 
              aes(x = x, y = y, label = label), 
              inherit.aes = FALSE, 
              color = "black", size = 4)
  
  print(p_time_sd)
}else{
  VL_times_df <- data.frame(rho = unique(AllResults$rho), VL_time = VL_times)
  
  VL_times_ann <- VL_times_df %>%
    mutate(
      x = 1.3,
      y = max(AllResults$time) - 8,
      label = paste0("Cholesky: ", signif(VL_time, 3), " s") 
    )
  if(variance){
    variance_data <- AllResults %>%
      group_by(rho, preconditioner, t) %>%
      summarise(variance = var(negLL_diff, na.rm = TRUE), .groups = "drop")
    
    variance_data$t <- as.factor(variance_data$t)
    
    p_diff <- ggplot(variance_data, aes(x = t, y = variance, color = preconditioner)) +
      stat_summary(aes(group = preconditioner), fun = mean, geom = 'line', linewidth = 1) +
      stat_summary(aes(group = preconditioner), fun = mean, geom = 'point', size = 2) +
      facet_wrap(~rho, nrow = 1, scales = "fixed",
                 labeller = label_bquote(rho == .(rho))) +
      scale_y_log10() +
      scale_color_brewer(type = "qual", palette = 6,
                         labels = preconditioner_labels) +
      scale_shape_manual(values = seq_along(PRECONDITIONER), labels = scales::parse_format()) +
      theme_bw() +
      theme(
        axis.title.x = element_blank(),
        text = element_text(size = 15),
        legend.position = "top",
        axis.title.y = element_text(margin = margin(r = 10)),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()
      ) +
      ylab("Variance diff to Cholesky")
  }else{
    # Boxplot of difference to Cholesky
    p_diff <- ggplot(AllResults, aes(x=t, y=negLL_diff, fill=preconditioner)) +
      geom_hline(yintercept=0, linetype = "dashed") +
      geom_boxplot(outlier.size = 1) +
      facet_wrap(~rho, nrow = 1, scales = "fixed",
                 labeller = label_bquote(rho == .(rho))) +
      scale_fill_brewer(type = "qual", palette = 6,
                        labels = preconditioner_labels) +
      theme_bw() +
      theme(
        axis.title.x=element_blank(),
        text = element_text(size=15),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = "top",
        axis.title.y = element_text(margin = margin(r = 10))
      ) +
      guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
      ylab("Difference to Cholesky")
  }
  
  
  # Timing plot
  p_time <- ggplot(AllResults, aes(x=t, y=time, color=preconditioner, shape=preconditioner)) +
    stat_summary(aes(group = preconditioner), fun = mean, geom = 'line', linewidth=1) +
    stat_summary(aes(group = preconditioner), fun = mean, geom = 'point', size=2) +
    facet_wrap(~rho, nrow = 1, scales = "fixed",
               labeller = label_bquote(rho == .(rho))) +
    scale_color_brewer(type = "qual", palette = 6,
                       labels = preconditioner_labels) +
    scale_shape_manual(values = seq_along(PRECONDITIONER), labels = scales::parse_format()) +
    #scale_y_log10(limits = c(1, 100)) +  # Scala logaritmica per y
    theme_bw() +
    theme(
      text = element_text(size=15),
      legend.position = "none",
      axis.title.y = element_text(margin = margin(r = 10)),
      axis.title.x = element_text(margin = margin(t = 10)),
      strip.background = element_blank(),
      strip.text.x = element_blank()
    ) +
    xlab("Number of sample vectors") +
    ylab("Time (s)")
  
  
  p_time <- p_time +
    geom_text(data = VL_times_ann, 
              aes(x = x, y = y, label = label), 
              inherit.aes = FALSE, 
              color = "black", size = 4)
  
  grid.newpage()
  grid.draw(rbind(ggplotGrob(p_diff), ggplotGrob(p_time), size = "first"))
}
