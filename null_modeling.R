# null model tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
null_model <- null_model(mode = "classification") %>% 
  set_engine("parsnip")

# workflow ----
null_workflow <- workflow() %>% 
  add_model(null_model) %>% 
  add_recipe(recipe4)

# Create a cluster object and then register: 
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic.clearlog()
tic("Null Model")

null_tuned <- fit_resamples(null_workflow,
                           q_folds,
                           metrics = metric_set(roc_auc),
                           control = control_grid(save_pred = TRUE,
                                                       save_workflow = TRUE,
                                                       parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

null_tictoc <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(null_tuned, null_tictoc, file = "results/null.rda")