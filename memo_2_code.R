## Parth Patel Apr 29, new codes

library(tidyverse)
library(tidymodels)
library(naniar)

load("data/splits.rda")

## Qin Huang May 19, new codes

## missing views

t1 <- miss_var_summary(q_training)

view(t1)

missing_prop <- purrr::map_dbl(q_training, ~ mean(is.na(.)))

vars_to_remove <- names(missing_prop[missing_prop > 0.2])

## recipe 1: most basic

recipe1 = recipe(QI1 ~ ., data = q_training) %>%
  step_string2factor(all_nominal()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal())


## recipe 2: adding step_other, step_nzv

recipe2 = recipe(QI1 ~ ., data = q_training) %>%
  step_string2factor(all_nominal()) %>%
  step_other(all_nominal(), -all_outcomes(), threshold = 0.05) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal()) 

## recipe 3: basic with imputation after removing all over 20%

recipe3 = recipe(QI1 ~ ., data = q_training) %>%
  step_impute_knn(all_predictors()) %>%
  step_string2factor(all_nominal()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal())

recipe3 %>%
  prep(q_training) %>% 
  bake(new_data = NULL)

## recipe 4: advanced with imputation used in MEMO 2

recipe4 <- recipe(QI1 ~ ., data = q_training) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())

recipe4 %>%
  prep(q_training) %>% 
  bake(new_data = NULL)


## recipe 5: advanced with imputation used in MEMO 2
recipe5 <- recipe(QI1 ~ ., data = q_training) %>% 
  step_impute_bag(all_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())

recipe5 %>%
  prep(q_training) %>% 
  bake(new_data = NULL)

save(recipe4, recipe5, file = "data/recipes.rda")

# We have two recipes at hand. The basic one has four steps: 'step_string2factor' is used to convert character variables to factors; ' step_dummy' is used to convert factor variables into a series of binary (0 and 1) variables; ' step_center'is used to center variables, which means subtracting the mean of a variable from all its values;  'step_scale' is used to scale variables, which means dividing all the values of a variable by its standard deviation. This basic recipe also removes variables with over 20% missing values and uses knn for imputation through 'step_rm' and 'step_impute_knn'.

# The second recipe goes beyond the basic by adding two more steps: 'step_other' is used to collapse infrequent factor levels into a single "other" level; 'step_nzv'is used to identify and remove predictors that have near-zero variance.


# We have four recipes at hand. The basic one has four steps: 'step_string2factor' is used to convert character variables to factors; ' step_dummy' is used to convert factor variables into a series of binary (0 and 1) variables; ' step_center'is used to center variables, which means subtracting the mean of a variable from all its values;  'step_scale' is used to scale variables, which means dividing all the values of a variable by its standard deviation. 

# The second recipe goes beyond the basic by adding two more steps: 'step_other' is used to collapse infrequent factor levels into a single "other" level; 'step_nzv'is used to identify and remove predictors that have near-zero variance.

# The third recipe and fourth recipe build on the first and second one by removing variables with over 20% missing values and using knn for imputation through 'step_rm' and 'step_impute_knn'.