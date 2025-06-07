library(tidyverse)
library(textrecipes)
library(tidymodels)
library(tidytext)
library(stylo)
library(stopwords)
library(embed)
library(future)
library(baguette)
library(discrim)
library(earth)
library(mda)

corpus_stylo <- load.corpus.and.parse(corpus.dir = "./corpus")

corpus_samples1 <- make.samples(corpus_stylo, 
                               sample.size = 2000, 
                               sampling = "normal.sampling",
                               sample.overlap = 0,
                               sampling.with.replacement = FALSE)

mfw_1 <- make.frequency.list(corpus_samples, head = 1000)
names <- c('lovelace', 'jones', 'tom', 'george', 'phineas', 'john', 'maggie', 
           'arthur', 'laura', 'adam', 'jane', 'harlowe', 'toby', 'howe', 'crawley',
           'lydgate', 'dorothea', 'pendennis', 'mary', 'joseph', 'belford',
           'sophia', 'emma', 'clarissa', 'pamela', 'micawber', 'peggotty', 'adams',
           'hetty', 'lucy', 'tulliver', 'elinor', 'elizabeth', 'jack', 'casaubon', 
           'james', 'amelia', 'osborne', 'violet', 'richard', 'fred', 'fanny',
           'bulstrode', 'rosamond', 'marianne', 'harriet', 'solmes', 'rose')
other <- c('ll', 've', 'em', 'de', 'st')

stopwords <- stop_words$word

mfw_tibble1 <- as_tibble(mfw_1)
mfw_tibble1 <- mfw_tibble1 |>  
  filter(!value %in% stopwords) |>  
  filter(!value %in% names) 

mfw_cleaned1 <- mfw_tibble1 |> 
  pull(value)

corpus_tf1 <- stylo::make.table.of.frequencies(corpus_samples1, mfw_cleaned1) |> 
  as.data.frame.matrix() |> 
  rownames_to_column("id") |> 
  as_tibble()

corpus_tf1 <- corpus_tf1 |> 
  separate(id, into = c("author", "title", NA), sep = "_") 
corpus_tf1

corpus_tf1 <- corpus_tf1 |> 
  select(-title) 

set.seed(01062025)
data_split1 <- corpus_tf1 |> 
  mutate(author = as.factor(author)) |> 
  initial_split(strata = author)

data_train1 <- training(data_split1) 
data_test1 <- testing(data_split1)

set.seed(01062025)
folds1 <- vfold_cv(data_train1, strata = author, v = 5)
folds1

base_rec1 <- recipe(author ~ ., data = data_train1) |>
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors())

base_rec1

pca_rec1 <- base_rec1 |> 
  step_pca(all_predictors(), num_comp = tune())

pca1_rec

pls_rec1 <- base_rec1 |> 
  step_pls(all_numeric_predictors(), outcome = "author", num_comp = tune())

umap_rec1 <- base_rec1 |> 
  step_umap(all_numeric_predictors(), 
            outcome = "author",
            num_comp = tune(),
            neighbors = tune(),
            min_dist = tune()
  )

lasso_spec1 <- multinom_reg(penalty = tune(), mixture = 1) |> 
  set_mode("classification") |> 
  set_engine("glmnet")

ridge_spec1 <- multinom_reg(penalty = tune(), mixture = 0) |> 
  set_mode("classification") |> 
  set_engine("glmnet")

svm_spec1 <- svm_linear(cost = tune()) |> 
  set_mode("classification") |> 
  set_engine("LiblineaR")

mlp_spec1 <- mlp(hidden_units = tune(),
                penalty = tune(),
                epochs = tune()) |> 
  set_engine("nnet") |> 
  set_mode("classification")

fda_spec1 <- discrim_flexible(prod_degree = tune()) |> 
  set_engine("earth")

wflow_set1 <- workflow_set(  
  preproc = list(base = base_rec1,
                 pca = pca_rec1,
                 pls = pls_rec1,
                 umap = umap_rec1),  
  models = list(svm = svm_spec,
                lasso = lasso_spec,
                ridge = ridge_spec,
                mlp = mlp_spec,
                fda = fda_spec),  
  cross = TRUE
)

wflow_set1

plan(multisession, workers = 5)

train_res1 <- wflow_set1 |> 
  workflow_map(
    verbose = TRUE,
    seed = 180525,
    resamples = folds1,
    grid = 3,
    metrics = metric_set(f_meas, accuracy),
    control = control_resamples(save_pred = TRUE)
  )

plan(sequential)

autoplot(train_res1, metric = "accuracy") + 
  theme_light() +
  theme(legend.position = "none") +
  geom_text(aes(y = (mean - 2*std_err), label = wflow_id),
            angle = 90, hjust = 1.5) +
  coord_cartesian(ylim = c(-0.3, NA))

rank_results(train_res1, select_best = TRUE) |> 
  print()

autoplot(train_res1, id = "base_ridge") +
  theme_light()

best_results1 <- 
  train_res1 |> 
  extract_workflow_set_result("base_ridge") |> 
  select_best(metric = "accuracy")

print(best_results1)

ridge_res1 <- train_res1 |> 
  extract_workflow("base_ridge") |> 
  finalize_workflow(best_results1) |> 
  last_fit(split = data_split1, metrics = metric_set(f_meas, accuracy, roc_auc))

collect_metrics(ridge_res1) |> 
  print()

collect_predictions(ridge_res1) |> 
  conf_mat(truth = author, estimate = .pred_class) |> 
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "white", high = "#233857") +
  theme(panel.grid.major = element_line(colour = "#233857"),
        axis.text = element_text(color = "#233857"),
        axis.title = element_text(color = "#233857"),
        plot.title = element_text(color = "#233857"),
        axis.text.x = element_text(angle = 90))

collect_predictions(ridge_res1) |>
  roc_curve(truth = author, .pred_ABronte:.pred_Trollope) |>
  ggplot(aes(1 - specificity, sensitivity, color = .level)) +
  geom_abline(slope = 1, color = "gray50", lty = 2, alpha = 0.8) +
  geom_path(linewidth = 1.5, alpha = 0.7) +
  labs(color = NULL) +
  theme_light()

final_model1 <- extract_fit_parsnip(ridge_res1)

top_terms1 <- tidy(final_model1) |>
  filter(term != "(Intercept)") |>
  group_by(class) |>                           
  slice_max(abs(estimate), n = 7)  |>             
  ungroup()  |> 
  mutate(term = fct_reorder(term, abs(estimate)))

print(top_terms1)

top_terms1  |> 
  ggplot(aes(x = estimate, y = term, fill = class)) +
  geom_col(show.legend = FALSE, alpha = 0.85) +
  facet_wrap(~ class, scales = "free_y", nrow = 4) +
  labs(
    title = "Наиболее важные признаки для каждого автора",
    x = "Коэффициент",
    y = "Признак"
  ) +
  theme_minimal() 