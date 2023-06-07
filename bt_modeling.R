# boosted tree tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(xgboost)
library(doParallel)

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
bt_grid <- grid_regular(bt_param, levels = 8)

# workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(recipe4)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("Boosted Tree")

bt_tuned <- tune_grid(bt_workflow,
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

bt_tictoc <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(bt_tuned, bt_tictoc, file = "results/boosted_tree.rda")

load("results/boosted_tree.rda")

plot <- autoplot(bt_tuned)

save(plot, file = "plots/bt_r1.png")