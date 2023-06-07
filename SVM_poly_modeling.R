# support vector machine (polynomial) tuning ----

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
load("data/recipes.rda")

# Define model ----
svm_poly_model <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune(),
  mode = "classification"
) %>% 
  set_engine("kernlab")

# set-up tuning grid ----
svm_poly_param <- extract_parameter_set_dials(svm_poly_model)

# define tuning grid ----
svm_poly_grid <- grid_regular(svm_poly_param, levels = 8)

# workflow ----
svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(recipe4)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("SVM Polynomial")

svm_poly_tuned <- tune_grid(svm_poly_workflow,
                            resamples = q_folds,
                            grid = svm_poly_grid,
                            control = control_grid(save_pred = TRUE,
                                                   save_workflow = TRUE,
                                                   parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

svm_poly_tictoc <- tibble(model = log_time[[1]],
                          runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(svm_poly_tuned, svm_poly_tictoc, file = "results/svm_poly.rda")

load("results/svm_poly.rda")

plot <- autoplot(svm_poly_tuned)

save(plot, file = "plots/svm_poly_r1.png")
