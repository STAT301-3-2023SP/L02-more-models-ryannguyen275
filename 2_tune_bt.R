# Boosted Tree Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
bt_model <- boost_tree(mode = "classification",
                       mtry = tune(),
                       min_n = tune(),
                       learn_rate = tune()) %>% 
  set_engine("xgboost", importance = "impurity")


bt_param <- extract_parameter_set_dials(bt_model) %>% 
  update(mtry = mtry(range = c(1,13)))

bt_grid <- grid_regular(bt_param, levels = 5)

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("Boosted Tree")

bt_tuned <- tune_grid(bt_workflow,
                      resamples = wildfires_folds,
                      grid = bt_grid,
                      control = control_grid(save_pred = TRUE, 
                                             save_workflow = TRUE,
                                             parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

bt_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic)

save(bt_tuned, bt_tictoc, file = "results/bt_tuned.rda")