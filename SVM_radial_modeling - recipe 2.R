# support vector machine (radial) tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(kernlab)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/splits.rda")

# Define model ----
svm_radial_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune(),
  mode = "classification"
) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
svm_radial_param <- extract_parameter_set_dials(svm_radial_model)

# define tuning grid ----
svm_radial_grid <- grid_regular(svm_radial_param, levels = 8)

# workflow ----
svm_radial_workflow <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(recipe5)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("SVM Radial Recipe 2")

svm_radial_tuned_2 <- tune_grid(svm_radial_workflow,
                              resamples = q_folds,
                              grid = svm_radial_grid,
                              control = control_grid(save_pred = TRUE,
                                                     save_workflow = TRUE,
                                                     parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

svm_radial_tictoc_2 <- tibble(model = log_time[[1]],
                            runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(svm_radial_tuned_2, svm_radial_tictoc_2, file = "results/svm_radial_2.rda")

load("results/svm_radial_2.rda")

autoplot(svm_radial_tuned_2)
