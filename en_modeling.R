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

recipe4 = recipe(QI1 ~ ., data = q_training) %>%
  step_impute_knn(all_predictors())  %>%
  step_string2factor(all_nominal()) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal())

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
  add_recipe(recipe4)

# Tuning/fitting ----
doParallel::registerDoParallel(cores = 4)

tic.clearlog()
tic("Elastic Net")

en_tuned <- tune_grid(en_workflow,
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

en_tictoc <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(en_tuned, en_tictoc, file = "model_info/elastic_net.rda")