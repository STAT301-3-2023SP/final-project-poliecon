# elastic net tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(glmnet)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
en_model <- logistic_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "classification"
) %>% 
  set_engine("glmnet")

# set-up tuning grid ----
en_param <- extract_parameter_set_dials(en_model)

# define tuning grid ----
en_grid <- grid_regular(en_param, levels = 5)

# workflow ----
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(recipe5)

# Tuning/fitting ----
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

tic.clearlog()
tic("Elastic Net Recipe 2")

en_tuned_2 <- tune_grid(en_workflow,
                      resamples = q_folds,
                      grid = en_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

en_tictoc_2 <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(en_tuned_2, en_tictoc_2, file = "results/elastic_net_2.rda")