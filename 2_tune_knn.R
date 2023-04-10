# K-Nearest Neighbors Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
knn_model <- nearest_neighbor(mode = "classification",
                         neighbors = tune()) %>% 
  set_engine("kknn")

knn_param <- extract_parameter_set_dials(knn_model)

knn_grid <- grid_regular(knn_param, levels = 5)

knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("K-Nearest Neighbor")

knn_tuned <- tune_grid(knn_workflow,
                      resamples = wildfires_folds,
                      grid = knn_grid,
                      control = control_grid(save_pred = TRUE, 
                                             save_workflow = TRUE,
                                             parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(knn_tuned, knn_tictoc, file = "results/knn_tuned.rda")


