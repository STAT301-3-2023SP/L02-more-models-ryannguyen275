# SVM Radial Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
svm_radial_model <- svm_rbf(mode = "classification",
                            cost = tune(),
                            rbf_sigma = tune()) %>%
  set_engine("kernlab")

svm_radial_param <- extract_parameter_set_dials(svm_radial_model)

svm_radial_grid <- grid_regular(svm_radial_param, levels = 5)

svm_radial_workflow <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("SVM Radial")

svm_radial_tuned <- tune_grid(svm_radial_workflow,
                       resamples = wildfires_folds,
                       grid = svm_radial_grid,
                       control = control_grid(save_pred = TRUE, 
                                              save_workflow = TRUE,
                                              parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

svm_radial_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(svm_radial_tuned, svm_radial_tictoc, file = "results/svm_radial_tuned.rda")

