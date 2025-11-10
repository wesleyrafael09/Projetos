# ===============================================
# Análise de Dados - Ames Housing
# ===============================================

# Instalação e carregamento automático de pacotes
required_packages <- c("AmesHousing", "dplyr", "ggplot2", "skimr", "tidyr")
installed <- rownames(installed.packages())
for (pkg in setdiff(required_packages, installed)) {
  install.packages(pkg, dependencies = TRUE)
}
lapply(required_packages, library, character.only = TRUE)

# Definir diretórios fixos no seu computador
base_dir <- "C:....."
results_dir <- file.path(base_dir, "results")
data_dir <- file.path(base_dir, "data")

# Criar pastas se não existirem
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

cat("Pacotes carregados e pastas prontas em:\n", base_dir, "\n")

# Carregar o dataset Ames Housing (função make_ames)
cat("Carregando dataset Ames Housing...\n")
raw <- AmesHousing::make_ames()
cat(" ataset carregado:", nrow(raw), "linhas e", ncol(raw), "colunas.\n")

# Geração de resumo e exportação
skim <- skimr::skim(raw)
write.csv(as.data.frame(skim), file = file.path(results_dir, "skim_summary.csv"), row.names = FALSE)
cat(" Resumo estatístico salvo em", file.path(results_dir, "skim_summary.csv"), "\n")

# Seleção de variáveis principais
vars_keep <- c(
  "Sale_Price", "Gr_Liv_Area", "Overall_Qual", "Year_Built",
  "Full_Bath", "Half_Bath", "TotRms_AbvGrd", "Garage_Cars", "Lot_Area"
)

ames_small <- raw %>%
  select(all_of(vars_keep)) %>%
  rename(
    sale_price = Sale_Price,
    gr_liv_area = Gr_Liv_Area,
    overall_qual = Overall_Qual,
    year_built = Year_Built,
    full_bath = Full_Bath,
    half_bath = Half_Bath,
    tot_rooms = TotRms_AbvGrd,
    garage_cars = Garage_Cars,
    lot_area = Lot_Area
  ) %>%
  drop_na() %>%
  mutate(house_age = 2025 - year_built)

# Estatísticas descritivas básicas
desc <- ames_small %>%
  summarise(
    n = n(),
    mean_price = mean(sale_price),
    median_price = median(sale_price),
    sd_price = sd(sale_price)
  )
write.csv(desc, file = file.path(results_dir, "desc_stats.csv"), row.names = FALSE)
cat(" Estatísticas básicas salvas em", file.path(results_dir, "desc_stats.csv"), "\n")

#  Gráficos interativos + salvos em PNG
cat(" Gerando gráficos...\n")

# Histograma
g1 <- ggplot(ames_small, aes(x = sale_price)) +
  geom_histogram(bins = 50, fill = "skyblue", color = "white") +
  labs(title = "Histograma: Sale Price", x = "Preço de venda", y = "Frequência") +
  theme_minimal()

# Dispersão (preço x área)
g2 <- ggplot(ames_small, aes(x = gr_liv_area, y = sale_price)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "darkred") +
  labs(title = "Sale Price vs Ground Living Area", x = "Área (Living)", y = "Preço de Venda") +
  theme_minimal()

# Salvar e exibir os gráficos
ggsave(file.path(results_dir, "price_hist.png"), g1, width = 10, height = 6)
ggsave(file.path(results_dir, "price_vs_area.png"), g2, width = 10, height = 6)

print(g1)
print(g2)

# Salvar dataset tratado
saveRDS(ames_small, file = file.path(data_dir, "ames_small.rds"))
cat("Dados preparados e salvos em", file.path(data_dir, "ames_small.rds"), "\n")

# Visualização interativa do dataset
cat(" Abrindo dataset limpo para visualização...\n")
View(ames_small)

cat("\n script finalizado com sucesso!\n")
cat("Arquivos salvos em:\n")
cat(" -", file.path(results_dir, "skim_summary.csv"), "\n")
cat(" -", file.path(results_dir, "desc_stats.csv"), "\n")
cat(" -", file.path(results_dir, "price_hist.png"), "\n")
cat(" -", file.path(results_dir, "price_vs_area.png"), "\n")
cat(" -", file.path(data_dir, "ames_small.rds"), "\n")

