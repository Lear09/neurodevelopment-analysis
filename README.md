# neurodevelopment-analysis
Statistical analysis of pesticide exposure and neurodevelopmental outcomes — ELFE cohort

# Neurodevelopment Analysis - ELFE cohort
Statistical analysis of prenatal and postanatal pesticide exposure and neurodevelopmental outcomes in children from the ELFE French longitudinal birth cohort

# Author
Léa Ruch - Master's in Cognitive and Integrative Neurosciences,
University of Toulouse

# Research context
These scripts were developed during a research internship at the Luxembourg Institute of Health (LIH), Human Biomonitoring Research Unit, under the supervision of Dr. Brice Appenzeller and Dr. Linda Macheka

# Scripts 
## elastic_net_neurodevelopment.py
Elastic Net regression to identify pesticide predictors of developmental quotient (QD) in children at 3.5 years
Methods: ElasticNetCV, bootstrap stability selection
Visualization: coefficient paths, dot plot, stability plot
## PLSDA_neurodevelopment.py
PLS-DA to classify children with and without neurodevelopmental disorders based on pesticide exposure profiles
## linear_regression_forest_plot.py
Adjusted linear regression for each pesticide predicting developmental quotient (QD) at 3.5 years. Outputs forest plot and Excel results table
## BKMR_mixture_analysis.R
Bayesian Kernel Machine Regression (BKMR) to estimate joint effects of pesticide mixtures on neurodevelopment. Includes PIPs, overall risk, single exposure effects and interactions

# Methods used
WQS, BKMR, PLS-DA, Elastic Net, PCA, logistic and linear regression, Spearman correlation

# Note
Data not included for confidentiality reasons (ELFE cohort)
