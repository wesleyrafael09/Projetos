# ===============================================
# 3_shiny_app/app.R — Interface Shiny para previsão de preços
# ===============================================

# Pacotes necessários
required_pkgs <- c("shiny", "ggplot2", "dplyr", "DT", "randomForest", "caret")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)
lapply(required_pkgs, library, character.only = TRUE)

# ===============================================
# Diretórios fixos
# ===============================================
base_dir <- "C:......."
data_dir <- file.path(base_dir, "data")
models_dir <- file.path(base_dir, "models")

# ===============================================
# Carregar dados e modelos salvos
# ===============================================
ames_small <- readRDS(file.path(data_dir, "ames_small.rds"))
lm_fit <- readRDS(file.path(models_dir, "lm_model.rds"))
rf_fit <- readRDS(file.path(models_dir, "rf_model.rds"))
preproc <- readRDS(file.path(models_dir, "preproc.rds"))

cat(" Dados e modelos carregados com sucesso.\n")

# ===============================================
# Interface do aplicativo (UI)
# ===============================================
ui <- fluidPage(
  titlePanel(" Previsão de Preços - Ames Housing"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("gr_liv_area", "Ground Living Area (sq ft):", value = 1500, min = 200, max = 10000, step = 10),
      sliderInput("overall_qual", "Overall Quality:", min = 1, max = 10, value = 6),
      numericInput("year_built", "Year Built:", value = 1990, min = 1800, max = 2025),
      numericInput("full_bath", "Full Baths:", value = 2, min = 0, max = 10),
      numericInput("half_bath", "Half Baths:", value = 0, min = 0, max = 10),
      numericInput("tot_rooms", "Total Rooms Above Grade:", value = 6, min = 1, max = 20),
      numericInput("garage_cars", "Garage Cars:", value = 2, min = 0, max = 5),
      numericInput("lot_area", "Lot Area:", value = 8000, min = 200, max = 100000),
      actionButton("predict", "Prever preço", class = "btn-primary")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel(" Gráficos", plotOutput("price_vs_area")),
        tabPanel(" Previsão", verbatimTextOutput("prediction")),
        tabPanel(" Dados (amostra)", DT::dataTableOutput("table"))
      )
    )
  )
)

# ===============================================
# Servidor (Server)
# ===============================================
server <- function(input, output, session) {
  
  # Gráfico básico de dispersão
  output$price_vs_area <- renderPlot({
    ggplot(ames_small, aes(x = gr_liv_area, y = sale_price)) +
      geom_point(alpha = 0.4, color = "steelblue") +
      geom_smooth(method = "lm", se = TRUE, color = "darkred") +
      labs(title = "Preço de Venda vs Área Construída",
           x = "Área (Living Area)", y = "Preço de Venda") +
      theme_minimal()
  })
  
  # Amostra de dados
  output$table <- DT::renderDataTable({
    DT::datatable(head(ames_small, 200))
  })
  
  # Ação de previsão
  observeEvent(input$predict, {
    newdata <- tibble(
      gr_liv_area = input$gr_liv_area,
      overall_qual = input$overall_qual,
      year_built = input$year_built,
      full_bath = input$full_bath,
      half_bath = input$half_bath,
      tot_rooms = input$tot_rooms,
      garage_cars = input$garage_cars,
      lot_area = input$lot_area
    ) %>%
      mutate(house_age = 2025 - year_built)
    
    # Aplicar normalização
    new_pp <- predict(preproc, newdata %>% select(-year_built))
    
    # Previsões
    pred_lm <- predict(lm_fit, newdata = new_pp)
    pred_rf <- predict(rf_fit, newdata = newdata)
    
    # Mostrar resultado
    output$prediction <- renderPrint({
      cat(" Previsões para os valores inseridos:\n\n")
      cat(" Linear Regression:\t$", round(pred_lm, 2), "\n")
      cat(" Random Forest:\t$", round(pred_rf, 2), "\n")
    })
  })
}

# ===============================================
# Executar o app
# ===============================================
shinyApp(ui, server)


