# rand forest tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(ranger)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/.rda")

### ADD RECIPE HERE

# knn_impute_rec %>% prep() %>% bake(new_data = NULL)

# Define model ----
rf_model <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  mode = "classification"
) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_param <- extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(c(1, #Change to Variable Amount)))

# define tuning grid ----
rf_grid <- grid_regular(rf_param, levels = 5)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bag_impute_rec)

# Tuning/fitting ----
doParallel::registerDoParallel(cores = 8)

tic.clearlog()
tic("Random Forest")

rf_tuned <- tune_grid(rf_workflow,
                          resamples = titanic_folds,
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