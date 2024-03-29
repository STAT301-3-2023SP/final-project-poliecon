# single layer neural network (mlp) tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(nnet)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  mode = "classification"
) %>% 
  set_engine("nnet")

# set-up tuning grid ----
nn_param <- extract_parameter_set_dials(nn_model)

# define tuning grid ----
nn_grid <- grid_regular(nn_param, levels = 8)

# workflow ----
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(recipe4)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("Single Layer Neural Network")

nn_tuned <- tune_grid(nn_workflow,
                      resamples = q_folds,
                      grid = nn_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

nn_tictoc <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(nn_tuned, nn_tictoc, file = "results/slnn_mlp.rda")

load("results/slnn_mlp.rda")

autoplot(nn_tuned)