# nearest neighbor tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(kknn)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
kn_model <- nearest_neighbor(
  neighbors = tune(),
  mode = "classification"
) %>% 
  set_engine("kknn")

# set-up tuning grid ----
kn_param <- extract_parameter_set_dials(kn_model)

# define tuning grid ----
kn_grid <- grid_regular(kn_param, levels = 8)

# workflow ----
kn_workflow <- workflow() %>% 
  add_model(kn_model) %>% 
  add_recipe(recipe4)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("K-Nearest Neighbor")

kn_tuned <- tune_grid(kn_workflow,
                      resamples = q_folds,
                      grid = kn_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

kn_tictoc <- tibble(model = log_time[[1]],
                    runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(kn_tuned, kn_tictoc, file = "results/knn.rda")

load("results/knn.rda")

autoplot(kn_tuned)