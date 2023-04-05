# Random Forest Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
rf_model <- rand_forest(min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

rf_param <- extract_parameter_set_dials(rf_model)

rf_grid <- grid_regular(rf_param, levels = 5)

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("Random Forest")

rf_tuned <- tune_grid(rf_workflow,
                       resamples = wildfires_folds,
                       grid = rf_grid,
                       control = control_grid(save_pred = TRUE, 
                                              save_workflow = TRUE,
                                              parallel_over = "everything"),
                       metrics = metrics)

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)


