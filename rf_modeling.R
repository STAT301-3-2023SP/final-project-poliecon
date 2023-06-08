# rand forest tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(ranger)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
rf_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  mode = "classification"
) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(range = c(1,22)))

# define tuning grid ----
rf_grid <- grid_regular(rf_params, levels = 8)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe4)


# Create a cluster object and then register: 
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("Random Forest")

rf_tuned <- tune_grid(rf_workflow,
                          resamples = q_folds,
                          grid = rf_grid,
                          control = control_grid(save_pred = TRUE,
                                                 save_workflow = TRUE,
                                                 parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

rf_tictoc <- tibble(model = log_time[[1]],
                        runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(rf_tuned, rf_tictoc, file = "results/rf.rda")

load("results/rf.rda")

autoplot(rf_tuned)
