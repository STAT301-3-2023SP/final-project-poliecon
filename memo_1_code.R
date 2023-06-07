## Parth Patel Apr 29, new codes

library(tidyverse)
library(tidymodels)
library(naniar)
library(dataMaid)

load("data/raw/35012-0001-Data.rda")

data <- da35012.0001

## response variable

table(data$QI1)

## predictors

#data1 <- data %>% 
#  select(QI1, QIII1, QIII20, QV1, QV2,
 #        QVI1, QVI2, QVI3, QVI4, QVI6, QVI17, QVI22,
  #       QVI23, QVI24, QVI25, QVI_30, QVI_31, QVI_32,
   #      QVII1, QVII2, QVII3, QVII4, QVII5, QVII6, QVII7, QVII8, QVII9,
    #     QIX, QIX2, QIX3, QIX4, QIX5, QIX6, QIX7, QIX8, QIX9,
     #    QX2, QX7,QX8,QX9,QX10,QX11,QX12,QX13,QX14,QX16,QX17,
      #   QX18, QX20, QX21, QX22, QX23
 # ) %>% 
  #mutate(QI1 = as_factor(QI1)) %>% 
  #filter(!is.na(QI1)) 


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

makeDataReport(data2)

t1 <- miss_var_summary(data2)

view(t1)

## splits & folds

q_split <- initial_split(data2, prop = 0.8, strata = QI1)
q_training <- training(q_split)
q_test <- testing(q_split)

q_folds <- vfold_cv(q_training, v = 5, repeats = 3, strata = QI1)

save(q_training, q_test, q_folds, file = "data/splits.rda")