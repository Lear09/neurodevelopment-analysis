# ======================================
Elastic Net Regression - Pesticide exposure and neurodevelopmental outcomes (ELFE cohort)
Author: Léa Ruch 
Description: Elastic Net with cross validation to identify pesticide predictors of developmental quotient (QD) in children at 3.5 years. Includes coefficient paths, dot plot of selected variables, and bootstrap stability selection.
Methods: ElasticNetCV, StandardScaler, bootstrap resampling (scikit-learn)
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# 1.Data

df = pd.read_excel("data/outcome_data.xlsx)  
df_pest = pd.read_excel("data/pesticide_data.xlsx)  
df = df.merge(df_pest, on="ID", how="inner")

outcome_var = "QD"
y = df[outcome_var].values

# 2️.Pesticides + covariables

pesticides_columns

covariables = ["Sex", "Mother age", "Mother education", "Father education", "Mother occupation", "Father occupation", "Birth weight", "Birth size", "Head circumference"]

X = df[pesticides_columns + covariables]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


# 3️.Standardize

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️.Elastic Net CV

model = ElasticNetCV(l1_ratio=0.5, alphas=np.logspace(-3, 1, 100), cv=5, max_iter=5000, random_state=42)
model.fit(X_scaled, y)

coef = pd.Series(model.coef_, index=X.columns)
coef_retained = coef[coef != 0].sort_values()

print("Variables retained by Elastic Net")
print(coef_retained)

# 5️.Coefficient Paths

coefs_path = []
for alpha in model.alphas_:
    enet = ElasticNetCV(l1_ratio=model.l1_ratio_, alphas=[alpha], cv=5, max_iter=5000)
    enet.fit(X_scaled, y)
    coefs_path.append(enet.coef_)

coefs_path = np.array(coefs_path)
plt.figure(figsize=(12,8))
for i in range(coefs_path.shape[1]):
    plt.plot(np.log10(model.alphas_), coefs_path[:, i], label=X.columns[i])
plt.xlabel("log10(Alpha)")
plt.ylabel("Coefficient (β)")
plt.title("Elastic Net Coefficient Paths (QD)")
plt.grid(False)
plt.tight_layout()
plt.show()


# 6️.Dot Plot of Selected Coefficients

coef_df = pd.DataFrame({'Variable': coef_retained.index, 'Coefficient': coef_retained.values})
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Coefficient',
    y='Variable',
    hue=(coef_df['Coefficient']>0),
    palette={True:'red', False:'blue'},
    size=abs(coef_df['Coefficient']),
    sizes=(50, 200),
    legend=False,
    data=coef_df
)
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel("Elastic Net Coefficient (β)")
plt.ylabel("Variable")
plt.title("Selected Variables by Elastic Net")
plt.tight_layout()
plt.show()


# 7️.Stability Selection via Bootstrap

n_boot = 100
selection_counts = pd.Series(0, index=X.columns)

for _ in range(n_boot):
    X_res, y_res = resample(X_scaled, y)
    enet = ElasticNetCV(l1_ratio=model.l1_ratio_, alphas=[model.alpha_], cv=5, max_iter=5000)
    enet.fit(X_res, y_res)
    selection_counts += (enet.coef_ != 0)

stability_df = pd.DataFrame({
    'Variable': X.columns,
    'SelectionFreq': selection_counts / n_boot * 100
}).sort_values('SelectionFreq', ascending=False)  

plt.figure(figsize=(12,14)) 
sns.barplot(
    x='SelectionFreq',
    y='Variable',
    data=stability_df,
    palette='viridis'  
)
plt.xlabel("Selection Frequency (%)")
plt.ylabel("Variable")
plt.title("Stability of Variable Selection (Bootstrap, QD)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils import resample

