# boosted tree tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(xgboost)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
bt_model <- boost_tree(
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune(),
  mode = "classification"
) %>% 
  set_engine("xgboost")

# set-up tuning grid ----
bt_param <- extract_parameter_set_dials(bt_model) %>% 
  update(mtry = mtry(c(1, 22)))

# define tuning grid ----
bt_grid <- grid_regular(bt_param, levels = 5)

# workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe5)

# Tuning/fitting ----
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic.clearlog()
tic("Boosted Tree Recipe 2")

bt_tuned_2 <- tune_grid(bt_workflow,
                      resamples = q_folds,
                      grid = bt_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

bt_tictoc_2 <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(bt_tuned_2, bt_tictoc_2, file = "results/boosted_tree_2.rda")