##============================-- 1. Set up Environment ----
## 1.1 load packages ----
rm(list=ls())

pkgs <- c("tidyverse", "tidymodels", "doParallel", "skimr", "bestNormalize")

lapply(pkgs, require, character.only = TRUE)

library(naniar)


##============================-- 2. Load Data and Preprocess ----
## 2.1 load Data ----


load("data/raw/35012-0001-Data.rda")

data <- da35012.0001


data1 <- data %>% 
  select(QI1, QVI6, QVI17, QVI22,
         QVI23, QVI_30, QVI_31, QVI_32,
         QVII1, QVII2, QVII3, QVII4, QVII7, QVII8, QVII9,
         QIX, QIX2, QIX3, QIX4, QIX5, QIX6,QIX9,
         QX2, QX7,QX9,QX11,QX17, QX22, QX23
  ) %>% 
  mutate(QI1 = as_factor(QI1)) %>% 
  filter(!is.na(QI1)) 

missing_prop <- purrr::map_dbl(data1, ~ mean(is.na(.)))

vars_to_remove <- names(missing_prop[missing_prop > 0.2])

data2 <- data1 %>%
  select(-all_of(vars_to_remove), -QVI6, -QX2, -QX11, -QX7)


t1 <- miss_var_summary(data2)

## 2.2 splits & folds ---

q_split <- initial_split(data2, prop = 0.8, strata = QI1)
q_training <- training(q_split)
q_test <- testing(q_split)

q_folds <- vfold_cv(q_training, v = 5, repeats = 3, strata = QI1)

save(q_training, q_test, q_folds, file = "data/splits.rda")


## 2.2 EDA ----
### 2.2.1 Label distribution ----


q_training %>%
  mutate(QI1 =as.character(QI1)) %>%
  count(QI1) %>%
  mutate(Prop = n/sum(n)) 


q_training %>%
  mutate(QI1 =as.character(QI1)) %>%
  count(QI1) %>%
  mutate(Prop = n/sum(n)) %>%
  ggplot(aes(x="", y=Prop, fill=QI1)) +
  geom_bar(stat = "identity", width = 0.7) ## balanced label



### 2.2.2 overall data glance ----
glimpse(q_training)
skimr::skim_without_charts(q_training)


### missingness


gg_miss_var(q_training)

miss_var_summary(q_training)
