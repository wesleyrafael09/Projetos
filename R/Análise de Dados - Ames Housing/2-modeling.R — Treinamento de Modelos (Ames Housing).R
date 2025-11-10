# ===============================================
# 2_modeling.R — Treinamento de Modelos (Ames Housing)
# ===============================================

# Instalação e carregamento de pacotes necessários
required_pkgs <- c("tidyverse", "caret", "randomForest", "e1071")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)

lapply(required_pkgs, library, character.only = TRUE)

# ===============================================
# Definir diretórios fixos
# ===============================================
base_dir <- "C:...."
data_dir <- file.path(base_dir, "data")
results_dir <- file.path(base_dir, "results")
models_dir <- file.path(base_dir, "models")

# Criar pastas se não existirem
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

cat(" Diretórios configurados com sucesso.\n")

# ===============================================
# Carregar os dados preparados
# ===============================================
ames_small <- readRDS(file.path(data_dir, "ames_small.rds"))
cat(" Dados carregados com", nrow(ames_small), "linhas e", ncol(ames_small), "colunas.\n")

# ===============================================
# Divisão treino/teste
# ===============================================
set.seed(123)
train_index <- createDataPartition(ames_small$sale_price, p = 0.8, list = FALSE)
train <- ames_small[train_index, ]
test <- ames_small[-train_index, ]

# ===============================================
# Fórmula do modelo
# ===============================================
formula <- sale_price ~ gr_liv_area + overall_qual + house_age +
  full_bath + half_bath + tot_rooms + garage_cars + lot_area

# ===============================================
# Normalização dos dados
# ===============================================
preproc <- preProcess(train %>% select(-sale_price), method = c("center", "scale"))
train_pp <- predict(preproc, train %>% select(-sale_price))
train_pp$sale_price <- train$sale_price

test_pp <- predict(preproc, test %>% select(-sale_price))
test_pp$sale_price <- test$sale_price

# ===============================================
# 1) Modelo de Regressão Linear
# ===============================================
lm_fit <- lm(formula, data = train_pp)
summary(lm_fit)

pred_lm <- predict(lm_fit, newdata = test_pp)
rmse_lm <- sqrt(mean((pred_lm - test_pp$sale_price)^2))
mae_lm <- mean(abs(pred_lm - test_pp$sale_price))

# ===============================================
# 2) Modelo de Random Forest
# ===============================================
set.seed(123)
rf_fit <- randomForest(formula, data = train, ntree = 200)

pred_rf <- predict(rf_fit, newdata = test)
rmse_rf <- sqrt(mean((pred_rf - test$sale_price)^2))
mae_rf <- mean(abs(pred_rf - test$sale_price))

# ===============================================
# Comparação dos resultados
# ===============================================
results <- tibble(
  model = c("Linear Regression", "Random Forest"),
  rmse = c(rmse_lm, rmse_rf),
  mae = c(mae_lm, mae_rf)
)

print(results)
write.csv(results, file = file.path(results_dir, "model_results.csv"), row.names = FALSE)

# ===============================================
# Salvar modelos treinados
# ===============================================
saveRDS(lm_fit, file = file.path(models_dir, "lm_model.rds"))
saveRDS(rf_fit, file = file.path(models_dir, "rf_model.rds"))
saveRDS(preproc, file = file.path(models_dir, "preproc.rds"))

cat("\n Modelos e resultados salvos com sucesso em:\n")
cat(" -", file.path(results_dir, "model_results.csv"), "\n")
cat(" -", file.path(models_dir, "lm_model.rds"), "\n")
cat(" -", file.path(models_dir, "rf_model.rds"), "\n")
cat(" -", file.path(models_dir, "preproc.rds"), "\n")

