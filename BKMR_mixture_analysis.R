BKMR Analysis - Pesticide mixture and neurodevelopement 
Author: Léa Ruch 
Description: Bayesian Kernel Machine Regression to estimate the joint effect of pesticide mixtures on developmental quotient (QD) in the ELFE cohort: include PIPs, overall mixture effect, single exposure effects and interactions

rm(list = ls())

# 1.Packages


packages <- c(
  "bkmr",
  "readxl",
  "dplyr",
  "ggplot2",
  "tidyr"
)

for(pkg in packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}


# 2.Data

data <- read_excel("outcome_data.xlsx")

# 3.Variable

Outcome <- "QD"

Covariates <- c(
  "Sex",
  "Mother_age",
  "Maternal_education",
  "Smoking",
  "BMI"
)

Pesticides <- setdiff(
  names(data),
  c(
    "ID",
    Outcome,
    Covariates
  )
)

# 4.Matrices


Y <- data[[Outcome]]

X <- data %>%
  select(all_of(Covariates)) %>%
  as.matrix()

Z <- data %>%
  select(all_of(Pesticides)) %>%
  as.matrix()


# 5.Clean data


complete <- complete.cases(Y, X, Z)

Y <- Y[complete]

X <- X[complete,]

Z <- Z[complete,]


# 6.Standadize

X <- scale(X)

Z <- scale(Z)


# 7.Fit BKMR


set.seed(1234)

fit <- kmbayes(
  y = Y,
  Z = Z,
  X = X,
  iter = 30000,
  varsel = TRUE,
  verbose = TRUE
)

saveRDS(fit, "BKMR_model.rds")

# 8.Plots

pdf(
  "TracePlots.pdf",
  width = 10,
  height = 6
)

TracePlot(fit)

dev.off()


# 9.PIPs 

pips <- ExtractPIPs(fit)

write.csv(
  pips,
  "BKMR_PIPs.csv",
  row.names = FALSE
)

PIP_plot <- ggplot(
  pips,
  aes(
    x = reorder(variable,PIP),
    y = PIP
  )
)+

geom_col(fill="#0072B2")+

coord_flip()+

theme_classic(base_size=15)+

labs(
x="",
y="Posterior Inclusion Probability"
)

ggsave(
"PIP_plot.png",
PIP_plot,
width=7,
height=6,
dpi=300
)

# 10.Mixture effect

overall <- OverallRiskSummaries(
  fit = fit,
  qs = seq(
    0.25,
    0.75,
    by=0.05
  )
)

png(
"OverallRisk.png",
width=2000,
height=1800,
res=300
)

PlotOverallRiskSummaries(
overall
)

dev.off()

# 11.Single exposure effect

single <- SingVarRiskSummaries(
fit=fit
)

png(
"SingleExposureEffects.png",
width=2200,
height=2200,
res=300
)

PlotSingVarRiskSummaries(
single
)

dev.off()


# 12.Bivariate interactions


Top2 <- pips %>%
arrange(desc(PIP)) %>%
slice(1:2)

if(nrow(Top2)==2){

png(
"Interaction.png",
width=2000,
height=1800,
res=300
)

InteractPlot(
fit,
vars=c(
Top2$variable[1],
Top2$variable[2]
)
)

dev.off()

}

# 13.Risk table

OverallRisk <- OverallRiskSummaries(
fit,
qs = c(
0.25,
0.5,
0.75
)
)

write.csv(
OverallRisk,
"OverallRisk.csv",
row.names=FALSE
)


SingleRisk <- SingVarRiskSummaries(
fit
)

capture.output(
SingleRisk,
file="SingleRisk.txt"
)

capture.output(
summary(fit),
file="BKMR_summary.txt"
)

cat("=====================================\n")
cat("BKMR analysis completed successfully\n")
cat("=====================================\n")
cat("\nOutputs generated:\n\n")

cat("- BKMR_model.rds\n")
cat("- BKMR_PIPs.csv\n")
cat("- PIP_plot.png\n")
cat("- OverallRisk.png\n")
cat("- SingleExposureEffects.png\n")
cat("- Interaction.png\n")
cat("- OverallRisk.csv\n")
cat("- SingleRisk.txt\n")
cat("- BKMR_summary.txt\n")
