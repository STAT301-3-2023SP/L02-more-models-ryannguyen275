# MARS Tuning

##### LOAD PACKAGES/DATA ##############################################

library(tidymodels)
library(tidyverse)
library(tictoc)

library(doMC)
registerDoMC(cores = 4)

tidymodels_prefer ()

load("results/tuning_setup.rda")

##### DEFINE ENGINES/WORKFLOWS #########################################
mars_model <- mars(mode = "classification",
                   num_terms = tune(),
                   prod_degree = tune()) %>%
  set_engine("earth")

mars_param <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 13)))

mars_grid <- grid_regular(mars_param, levels = 5)

mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(wildfires_recipe1)

##### TUNE GRID #######################################################
tic.clearlog()
tic("MARS")

mars_tuned <- tune_grid(mars_workflow,
                       resamples = wildfires_folds,
                       grid = mars_grid,
                       control = control_grid(save_pred = TRUE, # create extra column for each prediction
                                              save_workflow = TRUE, # lets you use extract_workflow
                                              parallel_over = "everything"),
                       # metrics = metric_set())
)
toc(log = TRUE)
time_log <- tic.log(format = FALSE)

mars_tictoc <- tibble(
  model = time_log[[1]]$msg,
  #runtime = end time - start time
  runtime = time_log[[1]]$toc - time_log[[1]]$tic
)

save(mars_tuned, mars_tictoc, file = "results/mars_tuned.rda")


