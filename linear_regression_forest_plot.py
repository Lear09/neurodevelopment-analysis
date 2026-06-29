Adjusted Linear Regression + Forest Plot - Pesticide exposure and cognitive development
Author: Léa Ruch
Description: Multiple linear regression models for each pesticide predicting developmental quotient (QD) at 3.5 years, adjusted for sociodemographic covariates.
Methods: OLS regression (statsmodels), forest plot (matplotlib)
Visualization: forest plot and Excel results table


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 1.Data

neuro_file = r"data/outcome_data.xlsx"
pest_file  = r"data/pesticide_data.xlsx"

df_neuro = pd.read_excel(neuro_file)
df_pest  = pd.read_excel(pest_file)

df = df_neuro.merge(df_pest, on="ID", how="inner") # Merge ID

# Variables
outcome = "QD"
covariates = ["Sex", "Birth weight", "Birth size", "Head circumference", "Mother age", "Mother education", "Mother occupation", "Father education", "Father occupation"]
pesticides = [col for col in df_pest.columns if col != "ID"]


# 2.Multiple linear regression

rows = []

for p in pesticides:
    tmp = df[[outcome, p] + covariates].dropna()
    if len(tmp) < 10:  # sécurité
        continue

    X = sm.add_constant(tmp[[p] + covariates].astype(float))
    y = tmp[outcome].astype(float)
    model = sm.OLS(y, X).fit()

    beta = model.params[p]
    ci_low, ci_high = model.conf_int().loc[p]
    pval = model.pvalues[p]

    rows.append({
        "Pesticide": p,
        "β (adjusted)": beta,
        "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
        "p-value": "<0.001" if pval < 0.001 else f"{pval:.3f}",
        "n": int(model.nobs)
    })

results_df = pd.DataFrame(rows)
results_df["p_num"] = results_df["p-value"].replace("<0.001", "0.000").astype(float)
results_df = results_df.sort_values("p_num").reset_index(drop=True)


# 3. Dorest plot data

# IC
results_df["ci_low"] = results_df["95% CI"].str.extract(r"\[(.*?),")[0].astype(float)
results_df["ci_high"] = results_df["95% CI"].str.extract(r",(.*?)\]")[0].astype(float)

# p-value
def get_color(p):
    try:
        if p == "<0.001" or float(p) < 0.05:
            return "red"
        elif float(p) < 0.1:
            return "orange"
        else:
            return "black"
    except:
        return "black"

results_df["color"] = results_df["p-value"].apply(get_color)

# 4.Forest plot

plt.figure(figsize=(8, max(6, len(results_df)*0.25)))

for i, row in results_df.iterrows():
    plt.errorbar(
        x=row["β (adjusted)"],
        y=i,
        xerr=[[row["β (adjusted)"] - row["ci_low"]], [row["ci_high"] - row["β (adjusted)"]]],
        fmt='o',
        color=row["color"],
        ecolor=row["color"],
        capsize=3
    )
    # Ajouter p-value à côté
    plt.text(row["ci_high"] + 0.05, i, f"p={row['p-value']}", va='center', fontsize=8)


plt.yticks(range(len(results_df)), results_df["Pesticide"])
plt.axvline(0, color='black', linestyle='--')  # ligne de référence à 0
plt.xlabel("β (per log-unit increase)")
plt.title("Adjusted Associations Between Maternal Pesticides Exposure at Birth and Cognitive Development")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.grid(False)

plt.gca().invert_yaxis()  # premier pesticide en haut
plt.tight_layout()
plt.show()

pd.set_option('display.max_rows', None)
print(results_df[["Pesticide","β (adjusted)","95% CI","p-value","n"]])


# EXPORT EXCEL

final_table = results_df[
    ["Pesticide","β (adjusted)","95% CI","p-value","n"]
]

final_table.to_excel(
    "Adjusted_Regression_Results.xlsx",
    index=False
)

print("\nExcel file exported successfully.")
