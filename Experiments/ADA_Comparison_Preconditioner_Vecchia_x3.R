################################################################################
### Preconditioner Comparison (Vecchia) x3
################################################################################


#######
## Packages
#######

### Install/Load Packages
#source("https://raw.githubusercontent.com/TimGyger/iterativeFSA/refs/heads/main/Packages.R")
#install.packages("https://cran.r-project.org/src/contrib/Archive/RandomFields/RandomFields_3.3.14.tar.gz")

# Load the required libraries
library(fields)
library(dplyr)
library(ggpubr)
library(gpboost)
library(RandomFields)
library(RandomFieldsUtils)
library(ggplot2)
library(grid)

################################################################################
#Generate data

make_data <- function(n, 
                      likelihood = "bernoulli_logit",
                      sigma2=1, #marginal variance of GP
                      rho=0.1, #range parameter
                      cov_function="exponential",
                      matern_nu=2.5,
                      with_fixed_effects=FALSE,
                      beta,
                      gamma_shape = 1,
                      p) {
  
  #Simulate spatial Gaussian process
  coords <- matrix(runif(n*2),ncol=2)
  if (cov_function == "exponential"){
    RFmodel <- RMexp(var=sigma2, scale=rho)
  } else if(cov_function == "matern") {
    RFmodel <- RMmatern(matern_nu, notinvnu=TRUE, var=sigma2, scale=rho)
  } else {
    stop("cov_function not supported")
  }
  sim <- RFsimulate(RFmodel, x=coords)
  eps <- sim$variable1
  eps <- eps - mean(eps)
  if(with_fixed_effects){
    #   Simulate fixed effects
    X <- cbind(rep(1,n),matrix(runif(n*p)-0.5,ncol=p))
    f <- X %*% beta    
  } else{
    X <- NA
    f <- 0
  }
  b <- f+eps
  #Simulate response variable
  if (likelihood == "bernoulli_probit") {
    probs <- pnorm(b)
    y <- as.numeric(runif(n) < probs)
  } else if (likelihood == "bernoulli_logit") {
    probs <- 1/(1+exp(-(b)))
    y <- as.numeric(runif(n) < probs)
  } else if (likelihood == "poisson") {
    mu <- exp(b)
    y <- qpois(runif(n), lambda = mu)
  } else if (likelihood == "gamma") {
    y <- qgamma(runif(n), rate = gamma_shape * exp(-b), shape = gamma_shape) #E[y] = exp(b)
  } else if (likelihood == "gaussian") {
    mu <- b
    y <- rnorm(n,sd=0.05) + mu
  }
  list(y=y, coords=coords, b=b, X=X)
}

################################################################################
# Main simulation for multiple rho values
################################################################################
# Toy/Load/Save configuration
Toy <- FALSE
load <- TRUE
save <- TRUE

# Parameters
sigma2_true <- 1
n <- if (Toy) 10000 else 100000
n_rep <- if (Toy) 10 else 100
NN <- 20
rho_values <- c(0.25, 0.05, 0.01)

precond_map <- list(
  "Sigma_inv_plus_BtWB"                  = c("VADU",            "P[VADU]"),
  "piv_chol_on_Sigma"                    = c("Pivoted Cholesky","P[LRAC]"),
  "predictive_process_plus_diagonal_200" = c("FITC",            "P[FITC_200]"),
  "incomplete_cholesky"                  = c("ZIRC",            "P[ZIRC]"),
  "incomplete_cholesky_SN_B"             = c("ZIRC SN B",       "P[ZIRC_SN_B]"),
  "incomplete_cholesky_SN_A"             = c("ZIRC SN A",       "P[ZIRC_SN_A]"),
  "incomplete_cholesky_T"                = c("ZIRC T",          "P[ZIRC_T]"),
  "incomplete_cholesky_JM"               = c("ZIRC JM",         "P[ZIRC_JM]"),
  "incomplete_cholesky_TJM"              = c("ZIRC T + JM",     "P[ZIRC_TJM]"),
  "hlfpc"                                = c("HLF PC",          "P[HLFPC]"),
  "hlfpc_nystroem_last"                  = c("HLF PC N Last",   "P[HLFPCNL]"),
  "hlfpc_nystroem_random"                = c("HLF PC N Random", "P[HLFPCNR]"),
  "hlfpc_pivoted_cholesky"               = c("HLF PC PC Fact",  "P[HLFPCPC]"),
  "hlfpc_lanczos"                        = c("HLF PC Lanczos",  "P[HLFPCL]"),
  "hlfpc_rlra"                           = c("HLF PC RLRA",     "P[HLFPCRLRA]")
)

preconditioners_to_load <- c("Sigma_inv_plus_BtWB"
                             #,"piv_chol_on_Sigma"
                             #,"predictive_process_plus_diagonal_200"
                             #,"incomplete_cholesky_SN_B"
                             #,"incomplete_cholesky_SN_B"
                             #,"incomplete_cholesky_SN_A"
                             #,"incomplete_cholesky_T"
                             #,"incomplete_cholesky_JM"
                             #,"incomplete_cholesky_SN_A"
                             #,"incomplete_cholesky_TJM"
                             #,"incomplete_cholesky"
                             ,"hlfpc"
                             ,"hlfpc_nystroem_last"
                             #,"hlfpc_nystroem_random"
                             ,"hlfpc_rlra"
                             #,"hlfpc_pivoted_cholesky"
                             #,"hlfpc_lanczos"
                             )

if (load && Toy){
  AllResults_saved <- readRDS("AllResultsTOY.rds")
} else if(load){
  AllResults_saved <- readRDS("AllResults.rds")
}

# Vector of random trace sample sizes
NUM_RAND_VEC_TRACE <- c(10, 20, 50, 100)
# Preconditioners to test
PRECONDITIONER <- c("Sigma_inv_plus_BtWB"
                    , "piv_chol_on_Sigma"
                    , "predictive_process_plus_diagonal_200"
                    #,"incomplete_cholesky_SN_B"
                    #,"incomplete_cholesky_SN_A"
                    #,"incomplete_cholesky_T"
                    #,"incomplete_cholesky_JM"
                    #,"incomplete_cholesky_SN_A"
                    #,"incomplete_cholesky_TJM"
                    #,"incomplete_cholesky"
                    ,"hlfpc"
                    ,"hlfpc_nystroem_last"
                    #,"hlfpc_nystroem_random"
                    ,"hlfpc_rlra"
                    #,"hlfpc_pivoted_cholesky"
                    #,"hlfpc_lanczos"
)

AllResults <- data.frame()
VL_times <- numeric(0)
VL_results <- numeric(0)

for (rho_true in rho_values){
  if (load){
    loaded_itresults <- AllResults_saved %>%
      filter(rho == rho_true)
  }
  
  true_covpars <- c(sigma2_true, rho_true)
  
  cat("============================================================\n")
  cat("Running tests for rho =", rho_true, "\n")
  cat("============================================================\n")
  
  # Generate data
  set.seed(1)
  mydata <- make_data(n=n,
                      likelihood = "bernoulli_logit",
                      sigma2 = sigma2_true,
                      rho = rho_true,
                      cov_function="matern",
                      matern_nu=1.5,
                      with_fixed_effects=FALSE)
  
  #---------------------------------------------------------------------------
  # 1) Cholesky-based negLL for reference
  #---------------------------------------------------------------------------
  cat("Begin Cholesky computation...\n")
  
  if(load){
    VLresult <- loaded_itresults$negLL_VL[1]
    VLtime <- loaded_itresults$time_VL[1]
  }else{  
    VLmodel <- GPModel(gp_coords = mydata$coords[,1:2],
                              cov_function = "matern",
                              cov_fct_shape=1.5,
                              likelihood="bernoulli_logit",
                              matrix_inversion_method = "cholesky",
                              gp_approx = "vecchia",
                              vecchia_ordering = "random",
                              num_neighbors=NN,
                              num_parallel_threads = 16)
  
    VLmodel$set_optim_params(params = list(maxit=1,
                                         trace=TRUE))
    VLtime <- system.time(VLresult <- VLmodel$neg_log_likelihood(cov_pars=true_covpars, y=mydata$y))[3]
  }
  
  VL_times <- c(VL_times, VLtime)
  VL_results <- c(VL_results, VLresult)
  cat("Cholesky computation ended\n")
  
  #---------------------------------------------------------------------------
  # 2) Iterative approach with different preconditioners
  #---------------------------------------------------------------------------
  Itresults <- data.frame(matrix(nrow=length(NUM_RAND_VEC_TRACE)*length(PRECONDITIONER)*n_rep,ncol = 8))
  colnames(Itresults) <- c("preconditioner", "t", "negLL", "time", "rho", "negLL_diff", "negLL_VL", "time_VL")
  
  i <- 1
  for (p in seq_along(PRECONDITIONER)) {
    for (t in seq_along(NUM_RAND_VEC_TRACE)) {
      for (r in seq_len(n_rep)) {
        print(rho_true)
        print(p)
        print(t)
        print(r)
        
       
        
        # Setup parameters
        if(!load || !(PRECONDITIONER[p] %in% preconditioners_to_load)){
          Itmodel <- GPModel(gp_coords = mydata$coords[,1:2],
                             cov_function = "matern",
                             cov_fct_shape = 1.5,
                             likelihood = "bernoulli_logit",
                             matrix_inversion_method = "iterative",
                             gp_approx = "vecchia",
                             vecchia_ordering = "random",
                             num_neighbors = NN,
                             num_parallel_threads = 16)
          
          if (PRECONDITIONER[p] == "predictive_process_plus_diagonal_200") {
          # "predictive_process_plus_diagonal_200"
          piv_chol_rank <- if (Toy) 50 else 200
          Itmodel$set_optim_params(params = list(
            maxit = 1,
            trace = TRUE,
            num_rand_vec_trace = NUM_RAND_VEC_TRACE[t],
            cg_preconditioner_type = "predictive_process_plus_diagonal",
            seed_rand_vec_trace = (1000*r + i), # just a different seed
            piv_chol_rank = piv_chol_rank
          ))
          } else {
            Itmodel$set_optim_params(params = list(
              maxit = 1,
              trace = TRUE,
              num_rand_vec_trace = NUM_RAND_VEC_TRACE[t],
              cg_preconditioner_type = PRECONDITIONER[p],
              seed_rand_vec_trace = (1000*r + i)
            ))
          }
            
        }
        
        # Evaluate negative log-likelihood and timing
        if(load && (PRECONDITIONER[p] %in% preconditioners_to_load)){
          Itresults$time[i] <- loaded_itresults$time[i]
          Itresults$negLL[i] <- loaded_itresults$negLL[i]
        } else{
          Itresults$time[i] <- system.time(Itresults$negLL[i] <- Itmodel$neg_log_likelihood(cov_pars=true_covpars, y=mydata$y))[3]
        }
        print(Itresults$time[i])
        
        Itresults$preconditioner[i] <- PRECONDITIONER[p]
        Itresults$t[i] <- NUM_RAND_VEC_TRACE[t]
        Itresults$rho[i] <- rho_true
        
        i = i+1
        gc()
      }
    }
  }
  
  Itresults$negLL_diff <- Itresults$negLL - VL_results[length(VL_results)]
  Itresults$negLL_VL <- VL_results[length(VL_results)]
  Itresults$time_VL  <- VL_times[length(VL_times)]
  
  AllResults <- rbind(AllResults, Itresults)
}

if(save && Toy){
  saveRDS(AllResults, "AllResultsTOY.rds")
}else if(save){
  saveRDS(AllResults, "AllResults.rds")
}

################################################################################
# Create a data frame containing VL_times for each rho
VL_times_df <- data.frame(rho = unique(AllResults$rho), VL_time = VL_times)

VL_times_ann <- VL_times_df %>%
  mutate(
    x = 1.3,  
    y = max(AllResults$time) - 6,
    label = paste0("Cholesky: ", signif(VL_time, 3), " s") 
  )

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
