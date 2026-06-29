PLA-DA Analysis - Pesticide exposure and neurodevelopemental disorders (ELFE cohort)
Author: Léa Ruch
Description: Partial Least Squares Discriminant Analysis to classsify children with and without neurodevelopmental disorders (NDD) based on pesticide exposure
Methods: PLSRegression (scikit-learn), loading plots, stability analysis



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


# 1.Data

neuro_file = r"data/neurodevelopment_data.xlsx"
pest_file = r"data/pesticide_data.xlsx"

df_neuro = pd.read_excel(neuro_file)
df_pest = pd.read_excel(pest_file)

# Merge on ID
df = pd.merge(df_neuro, df_pest, on='ID', how='inner')

# 2.Variables

exclude_cols = ['ID', 'NND', 'TSA', 'langage', 'attention',
                'comportement', 'psy', 'macarthur_2y', 'cdi_3,5y',
               'RD_3,5y', 'birth size', 'birth weight', 'head circumference', 'sex', 'educ_mother', 'educ_father', 'pro_mother', 'pro_father']

X = df[[c for c in df.columns if c not in exclude_cols]].copy()  
X = X.apply(pd.to_numeric, errors='coerce').fillna(X.median())    

y = df['NND'].values  # 0 = Control, 1 = NND

# Standardize X
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3.PLS-DA

n_components = 2
pls = PLSRegression(n_components=n_components)
pls.fit(X_std, y)
X_scores = pls.x_scores_

explained_var = np.var(X_scores, axis=0) / np.sum(np.var(X_std, axis=0)) # % variance explained by each component (approximation)


# 4.Plot PLS-DA

plt.figure(figsize=(8,6))
colors = {0: 'blue', 1: 'red'}
labels = {0: 'Healthy', 1: 'NDD'}

for grp in [0,1]:
    idx = y == grp
    plt.scatter(X_scores[idx,0], X_scores[idx,1], color=colors[grp],
                label=labels[grp], alpha=0.7)

plt.xlabel(f'({explained_var[0]*100:.1f}%)')
plt.ylabel(f'({explained_var[1]*100:.1f}%)')
plt.title('PLS-DA')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()


# 6.Loading plot


loadings = pls.x_loadings_  # shape = (n_variables, n_components)
variables = X.columns

plt.figure(figsize=(10,8))

# Pour PLS1 et PLS2
plt.scatter(loadings[:,0], loadings[:,1], color='black', alpha=0.7)

for i, var in enumerate(variables):
    plt.text(loadings[i,0], loadings[i,1], var, fontsize=8, alpha=0.8)

plt.xlabel('PLS1 Loadings')
plt.ylabel('PLS2 Loadings')
plt.title('PLS-DA Loadings Plot (importance of variables)')
plt.grid(True)
plt.tight_layout()
plt.show()


#Loading bar plot (Top variables)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

comp_idx = 0  # (0 = PLS1, 1 = PLS2)

loadings = pls.x_loadings_[:, comp_idx] # loadings

loadings_df = pd.DataFrame({ # data frame
    'Variable': X.columns,
    'Loading': loadings
})

loadings_df['abs_Loading'] = loadings_df['Loading'].abs()

loadings_df = loadings_df.sort_values('abs_Loading', ascending=True)

top_n = 10
top_loadings = loadings_df.tail(top_n)

plt.figure(figsize=(8,6))
colors = ['red' if val > 0 else 'blue' for val in top_loadings['Loading']]  
plt.barh(top_loadings['Variable'], top_loadings['Loading'], color=colors)
plt.xlabel('PLS1 Loadings')
plt.title(f'Top {top_n} variables contributing to PLS1')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4.Loading for first component

loadings = pls.x_weights_[:, 0]  # PLS1
variables = X.columns

# Sort by absolute loading for visualization
loadings_df = pd.DataFrame({
    'Variable': variables,
    'PLS1_loading': loadings
}).assign(abs_loading=lambda df: df['PLS1_loading'].abs()) \
 .sort_values('abs_loading', ascending=True)  # ascending for horizontal barplot


# 5.Forest plot horizontal

plt.figure(figsize=(8, max(6, len(variables)*0.3))) 

# Color positive = red, negative = blue
colors = ['#d62728' if v > 0 else '#1f77b4' for v in loadings_df['PLS1_loading']]

plt.barh(loadings_df['Variable'], loadings_df['PLS1_loading'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('PLS1 Loading (Contribution to group separation)')
plt.title('Variable Contributions')
plt.tight_layout()
plt.show()


df["PLS1"] = X_scores[:, 0]
print(df.groupby("NND")["PLS1"].mean())
