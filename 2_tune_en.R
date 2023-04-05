# Elastic Net Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
en_model <- logistic_reg(mode = "classification",
                         penalty = tune(), 
                         mixture = tune()) %>% 
  set_engine("glmnet")

en_param <- extract_parameter_set_dials(en_model)

en_grid <- grid_regular(en_param, levels = 5)

# update recipe
wildfires_interact <- wildfires_recipe1 %>% 
  step_interact(~all_numeric_predictors()^2)

prep(wildfires_interact) %>% 
  bake(new_data = NULL)

en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(wildfires_interact)

##### TUNE GRID ########################################################
tic.clearlog()
tic("Elastic Net")

en_tuned <- tune_grid(en_workflow,
                         resamples = wildfires_folds,
                         grid = en_grid,
                         control = control_grid(save_pred = TRUE, 
                                                save_workflow = TRUE,
                                                parallel_over = "everything"))

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

en_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(en_tuned, en_tictoc, file = "results/en_tuned")

