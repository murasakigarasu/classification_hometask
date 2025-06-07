#Загружаем необходимые библиотеки для построения модели
library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stylo)
library(embed)
library(future)
library(baguette)
library(discrim)
library(earth)
library(mda)

#Загружаем корпус и сразу парсим при помощи функции из пакета stylo
corpus_stylo <- load.corpus.and.parse(corpus.dir = "./corpus")

#Делим романы на несколько текстов меньшей длины (по 2000 слов) 
corpus_samples <- make.samples(corpus_stylo, 
                               sample.size = 2000, 
                               sampling = "normal.sampling",
                               sample.overlap = 0,
                               sampling.with.replacement = FALSE)

#Берём 500 самых частотных токенов (в том числе и стоп-слова)
mfw <- make.frequency.list(corpus_samples)[1:500]

#Отдельно пропишем имена/фамилии героев, чтобы затем от них избавиться
names <- c('lovelace', 'jones', 'tom', 'george', 'phineas', 'john', 'maggie', 
           'arthur', 'laura', 'adam', 'jane', 'harlowe', 'howe')

mfw_tibble <- as_tibble(mfw)
mfw_tibble <- mfw_tibble |>  
  filter(!value %in% names)
mfw_cleaned <- mfw_tibble |> 
  pull(value)

#Составляем матрицу с частотностями
corpus_tf <- stylo::make.table.of.frequencies(corpus_samples, mfw_cleaned) |> 
  as.data.frame.matrix() |> 
  rownames_to_column("id") |> 
  as_tibble()

#Делим колонку id, чтобы получить имя автора
corpus_tf <- corpus_tf |> 
  separate(id, into = c("author", "title", NA), sep = "_") 
corpus_tf

#Смотрим распределение полчившихся отрывков по авторам
corpus_tf |> 
  count(author) |> 
  ggplot(aes(reorder(author, n), n, fill = author)) +
  geom_col(show.legend = FALSE) +
  xlab(NULL) +
  ylab(NULL) +
  scale_fill_viridis_d() + 
  theme_light() +
  coord_flip()

corpus_tf |> 
  count(author) |> 
  arrange(n)

#Удаляем колонку title
corpus_tf <- corpus_tf |>  
  select(-title) 

#Делим корпус на обучающую и тестовую выборки
set.seed(01062025)
data_split <- corpus_tf |> 
  mutate(author = as.factor(author)) |> 
  initial_split(strata = author)

data_train <- training(data_split) 
data_test <- testing(data_split)

set.seed(01062025)
folds <- vfold_cv(data_train, strata = author, v = 5)
folds

#Прописываем рецепты
base_rec <- recipe(author ~ ., data = data_train) |>
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors())

base_rec

pca_rec <- base_rec |> 
  step_pca(all_predictors(), num_comp = tune())

pca_rec

#Провожу разведывательный анализ
base_trained <- base_rec |>
  prep(data_train) 

base_trained

pls_trained <- base_trained |> 
  step_pls(all_numeric_predictors(), outcome = "author", num_comp = 5) |> 
  prep() 

pls_trained |> 
  juice() 

pls_trained |> 
  juice() |> 
  ggplot(aes(PLS1, PLS2, color = author)) +
  geom_point() +
  theme_light()

base_trained |> 
  step_umap(all_numeric_predictors(), outcome = "author", num_comp = 5) |> 
  prep() |> 
  juice() |> 
  ggplot(aes(UMAP1, UMAP2, color = author)) +
  geom_point(alpha = 0.5) +
  theme_light()

pls_rec <- base_rec |> 
  step_pls(all_numeric_predictors(), outcome = "author", num_comp = tune())

umap_rec <- base_rec |> 
  step_umap(all_numeric_predictors(), 
            outcome = "author",
            num_comp = tune(),
            neighbors = tune(),
            min_dist = tune()
  )

#Прописываем модели и методы
lasso_spec <- multinom_reg(penalty = tune(), mixture = 1) |> 
  set_mode("classification") |> 
  set_engine("glmnet")

ridge_spec <- multinom_reg(penalty = tune(), mixture = 0) |> 
  set_mode("classification") |> 
  set_engine("glmnet")

svm_spec <- svm_linear(cost = tune()) |> 
  set_mode("classification") |> 
  set_engine("LiblineaR")

mlp_spec <- mlp(hidden_units = tune(),
                penalty = tune(),
                epochs = tune()) |> 
  set_engine("nnet") |> 
  set_mode("classification")

fda_spec <- discrim_flexible(prod_degree = tune()) |> 
  set_engine("earth")

#Прописываем воркфлоу (с возможностью комбинацией рецептов и моделей)
wflow_set <- workflow_set(  
  preproc = list(base = base_rec,
                 pca = pca_rec,
                 pls = pls_rec,
                 umap = umap_rec),  
  models = list(svm = svm_spec,
                lasso = lasso_spec,
                ridge = ridge_spec,
                mlp = mlp_spec,
                fda = fda_spec),  
  cross = TRUE
)

wflow_set

#Параллелим вычисления
plan(multisession, workers = 5)

#Вычисляем...
train_res <- wflow_set |> 
  workflow_map(
    verbose = TRUE,
    seed = 180525,
    resamples = folds,
    grid = 3,
    metrics = metric_set(f_meas, accuracy),
    control = control_resamples(save_pred = TRUE)
  )

#Возвращаемся к изначальному последовательному вычислению
plan(sequential)

#Визуализируем полученные оценки моделей на графике
autoplot(train_res, metric = "accuracy") + 
  theme_light() +
  theme(legend.position = "none") +
  geom_text(aes(y = (mean - 2*std_err), label = wflow_id),
            angle = 90, hjust = 1.5) +
  coord_cartesian(ylim = c(-0.3, NA))

#Выбираем лучшую и дообучаем
rank_results(train_res, select_best = TRUE) |> 
  print()

autoplot(train_res, id = "base_ridge") +
  theme_light()

best_results <- 
  train_res |> 
  extract_workflow_set_result("base_ridge") |> 
  select_best(metric = "accuracy")

print(best_results)

ridge_res <- train_res |> 
  extract_workflow("base_ridge") |> 
  finalize_workflow(best_results) |> 
  last_fit(split = data_split, metrics = metric_set(f_meas, accuracy, roc_auc))

collect_metrics(ridge_res) |> 
  print()

#Создаем confusion matrix
collect_predictions(ridge_res) |> 
  conf_mat(truth = author, estimate = .pred_class) |> 
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "#233857") +
  theme(panel.grid.major = element_line(colour = "#233857"),
        axis.text = element_text(color = "#233857"),
        axis.title = element_text(color = "#233857"),
        plot.title = element_text(color = "#233857"),
        axis.text.x = element_text(angle = 90))

#строим roc-кривую
collect_predictions(ridge_res) |>
  roc_curve(truth = author, .pred_ABronte:.pred_Trollope) |>
  ggplot(aes(1 - specificity, sensitivity, color = .level)) +
  geom_abline(slope = 1, color = "gray50", lty = 2, alpha = 0.8) +
  geom_path(linewidth = 1.5, alpha = 0.7) +
  labs(color = NULL) +
  theme_light()

final_model <- extract_fit_parsnip(ridge_res)

#Собираем топ-слова для каждого автора и визуализируем
top_terms <- tidy(final_model) |>
  filter(term != "(Intercept)") |>
  group_by(class) |>                           
  slice_max(abs(estimate), n = 7)  |>             
  ungroup()  |> 
  mutate(term = fct_reorder(term, abs(estimate)))

print(top_terms)

top_terms  |> 
  ggplot(aes(x = estimate, y = term, fill = class)) +
  geom_col(show.legend = FALSE, alpha = 0.85) +
  facet_wrap(~ class, scales = "free_y", nrow = 4) +
  labs(
    title = "Наиболее важные признаки для каждого автора",
    x = "Коэффициент",
    y = "Признак"
  ) +
  theme_minimal() 








