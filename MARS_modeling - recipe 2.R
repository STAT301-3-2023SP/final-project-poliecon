# multivariate adaptive regression splines tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(earth)
library(doParallel)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(30123)

# load required objects ----
load("data/splits.rda")
load("data/recipes.rda")

# Define model ----
mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  mode = "classification"
) %>% 
  set_engine("earth")

# set-up tuning grid ----
mars_param <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(c(1, 22)))

# define tuning grid ----
mars_grid <- grid_regular(mars_param, levels = 8)

# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe5)

# Tuning/fitting ----
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

tic.clearlog()
tic("MARS Recipe 2")

mars_tuned_2 <- tune_grid(mars_workflow,
                        resamples = q_folds,
                        grid = mars_grid,
                        control = control_grid(save_pred = TRUE,
                                               save_workflow = TRUE,
                                               parallel_over = "everything")
)


# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
log_time <- tic.log(format = FALSE)

mars_tictoc_2 <- tibble(model = log_time[[1]],
                      runtime = log_time[[1]]$toc - log_time[[1]]$tic)

doParallel::stopImplicitCluster()

# Save
save(mars_tuned_2, mars_tictoc_2, file = "results/mars_2.rda")

load("results/mars_2.rda")

autoplot(mars_tuned_2)