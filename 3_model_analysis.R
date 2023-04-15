
library(tidyverse)
library(tidymodels)
library(kableExtra)
library(vip)
library(doMC)
library(parallel)

tidymodels_prefer()

registerDoMC(cores = 4)

##### LOAD-IN RESULTS ####################################################

result_files <- list.files("results/", "*.rda", full.names = TRUE)

for(i in result_files) {
  load(i)
}

load("results/tuning_setup.rda")

##### BASELINE/NULL MODEL ####################################################
null_mod <- null_model(mode = "classification") %>% 
  set_engine("parsnip")

null_wkflw <- workflow() %>% 
  add_model(null_mod) %>% 
  add_recipe(wildfires_recipe1)

null_fit <- null_wkflw %>% 
  fit_resamples(resamples = wildfires_folds,
                control = control_resamples(save_pred = TRUE))

null_fit %>% 
  collect_metrics()

##### ORGANIZE RESULTS TO GET BEST ###########################################

# Individual model results - tune_grid
# put in appendix

autoplot(en_tuned, metric = "roc_auc")

en_tuned %>% 
  show_best(metric = "roc_auc")


####### PUT ALL GRIDS TG ############
model_set <- as_workflow_set(
  "elastic_net" = en_tuned,
  "rand_forest" = rf_tuned, 
  "knn" = knn_tuned,
  "boosted_tree" = bt_tuned,
  "nn" = nn_tuned,
  "svm_poly" = svm_poly_tuned,
  "svm_radial" = svm_radial_tuned,
  "mars" = mars_tuned
)

##### plot of our results ###########

results_graph <- model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal() +
  geom_text(aes(y = mean - 0.03, label = wflow_id), angle = 90, hjust = 1) +
  ggtitle(label = "Best Results") +
  ylim(c(0.7, 0.9)) + 
  theme(legend.position = "none")

## Table of results
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% 
  select(best) %>% 
  unnest(cols = c(best))

## computation time

model_times <- bind_rows(en_tictoc,
                         bt_tictoc,
                         rf_tictoc,
                         knn_tictoc,
                         nn_tictoc,
                         svm_poly_tictoc,
                         svm_radial_tictoc,
                         mars_tictoc) %>% 
  mutate(wflow_id = c("elastic_net",
                      "boosted_tree",
                      "rand_forest",
                      "knn",
                      "nn",
                      "svm_poly",
                      "svm_radial",
                      "mars"))

result_table <- merge(model_results, model_times) %>% 
  select(model, mean, runtime) %>% 
  rename(roc_auc = mean)

result_table

nn_tuned %>% 
  show_best()

save(result_table, results_graph, file = "results/result_table.rda")

##### FINALIZE RESULTS #############################################

# finalize workflow
nn_workflow <- nn_workflow %>% 
  finalize_workflow(select_best(nn_tuned, metric = "roc_auc"))

# fit training data to final workflow 
fit_final <- fit(nn_workflow, wildfires_train)

# predict the testing data 
wildfires_pred_class <- predict(fit_final, wildfires_test) %>% 
  bind_cols(wildfires_test %>% select(wlf))

wildfires_pred <- wildfires_test %>%
  bind_cols(predict(fit_final, new_data = wildfires_test, type = "prob"))  %>%
  select(wlf, .pred_yes, .pred_no)

wildfires_pred

# final roc_auc 
roc_auc <- yardstick::roc_auc(wildfires_pred, truth = wlf, .pred_yes)

accuracy <- accuracy(wildfires_pred_class, wlf, .pred_class)

# confusion plot of results 
conf_matrix <- conf_mat(wildfires_pred_class, wlf, .pred_class)

conf_matrix

save(conf_matrix, accuracy, wildfires_pred_class, wildfires_pred, roc_auc, file = "results/final_results.rda")











