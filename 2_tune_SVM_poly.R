# SVM Polynomial Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
svm_poly_model <- svm_poly(mode = "classification",
                           cost = tune(),
                           degree = tune(),
                           scale_factor = tune()) %>%
  set_engine("kernlab")

svm_poly_param <- extract_parameter_set_dials(svm_poly_model)

svm_poly_grid <- grid_regular(svm_poly_param, levels = 5)

svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("SVM Polynomial")

svm_poly_tuned <- tune_grid(svm_poly_workflow,
                              resamples = wildfires_folds,
                              grid = svm_poly_grid,
                              control = control_grid(save_pred = TRUE, 
                                                     save_workflow = TRUE,
                                                     parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

svm_poly_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(svm_poly_tuned, svm_poly_tictoc, file = "results/svm_poly_tuned.rda")

