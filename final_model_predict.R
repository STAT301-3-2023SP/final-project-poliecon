library(tidyverse)
library(tidymodels)
library(gt)

load("plots/final_tables.rda")

ranked_mods

# the best model is the elastic net model with the first recipe (knn imputation)
# across the board, the first recipe was better for en, slnn, bt, and knn
# the second recipe (bag imp) was better for both svm models, mars, and rf

time_tib

# the elastic net model with the first recipe took 23.228 sec to run, making it one of the faster models

load("data/splits.rda")
load("results/model_set.rda")
load("results/elastic_net.rda")

best_results <- 
  model_set %>% 
  extract_workflow_set_result("elastic net recipe 2") %>% 
  select_best(metric = "roc_auc")

en_final_wf <- 
  model_set %>% 
  extract_workflow("elastic net recipe 2") %>% 
  finalize_workflow(best_results)

en_final_fit <- fit(en_final_wf, q_training)

final_pred <- predict(en_final_fit, new_data = q_test) %>% 
   bind_cols(q_test %>% select(QI1))

conf_matrix <- conf_mat(final_pred, QI1, .pred_class) %>% 
  autoplot(type = "heatmap")

save(final_pred, conf_matrix, file = "final_results/final_predict.rda")
