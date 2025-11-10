Este projeto cria uma aplicação interativa em R Shiny para explorar e prever preços de casas com base no conjunto de dados Ames Housing (versão reduzida: ames_small.rds).

Este projeto realiza uma **análise exploratória de dados (EDA)** no famoso dataset **Ames Housing**, amplamente usado em estudos de regressão e ciência de dados.

## O que o script faz

1. Instala e carrega automaticamente os pacotes necessários.  
2. Carrega o dataset `AmesHousing::make_ames()`.  
3. Seleciona variáveis relevantes (preço, área, qualidade, ano de construção etc).  
4. Limpa os dados e cria uma variável derivada `house_age`.  
5. Gera estatísticas descritivas e gráficos:
   - Histograma de preços (`price_hist.png`)
   - Dispersão entre área e preço (`price_vs_area.png`)
6. Salva o dataset limpo (`data/ames_small.rds`) e resumos em `results/`.

