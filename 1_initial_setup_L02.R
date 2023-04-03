# Load package(s)
library(tidymodels)
library(tidyverse)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## load data
wildfires_data <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-burned)

## visualize distribution, class imbalance
# if class imbalance, consider downsampling or upsampling

# missingness - none
# if present, impute; step_impute_mean _median _mode _knn _linear

# check for factor/categorical with MANY options (more than 10)

wildfires_data %>% 
  ggplot(aes(x = wlf)) +
  geom_bar()

skimr:: skim_without_charts(wildfires_data)

## split data into training/testing, create folds
set.seed(3013)
wildfires_split <- initial_split(wildfires_data, prop = 0.8, strata = wlf)

wildfires_train <- training(wildfires_split)

wildfires_test <- testing(wildfires_split)

wildfires_folds <- vfold_cv(wildfires_train, v = 5, repeats = 3, strata = wlf)

## recipes
wildfires_recipe1 <- recipe(wlf ~., data = wildfires_train) %>% 
  # accept new levels in testing data if not seen in training data
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>%  
  step_normalize(all_predictors())

prep(wildfires_recipe1) %>% 
  bake(new_data = NULL)

save(wildfires_recipe1, wildfires_train, wildfires_folds, wildfires_test,
     file = "results/tuning_setup.rda")

