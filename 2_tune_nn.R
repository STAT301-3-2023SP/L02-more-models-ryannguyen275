# Single Layer Neural Network Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
nn_model <- mlp(mode = "classification",
                hidden_units = tune(),
                penalty = tune()) %>%
  set_engine("nnet")

nn_param <- extract_parameter_set_dials(nn_model)

nn_grid <- grid_regular(nn_param, levels = 5)

nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID ########################################################
tic.clearlog()
tic("Neural Network")

nn_tuned <- tune_grid(nn_workflow,
                            resamples = wildfires_folds,
                            grid = nn_grid,
                            control = control_grid(save_pred = TRUE, 
                                                   save_workflow = TRUE,
                                                   parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

nn_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(nn_tuned, nn_tictoc, file = "results/nn_tuned.rda")