import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

results = pd.read_csv('./results.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

model_formula = 'forecast_mse ~ missing_rate * C(missing_type) * completeness_rate * C(imputation_method)'
model = ols(model_formula, data=results).fit()
anova_results = sm.stats.anova_lm(model, typ=3)

ss_total = model.ess + model.ssr
partial_r2 = pd.DataFrame({
    'Partial_R2': anova_results['sum_sq'] / ss_total
})
partial_r2 = partial_r2.sort_values('Partial_R2', ascending=False)
print(partial_r2)

model_formula = 'forecast_mae ~ missing_rate * C(missing_type) * completeness_rate * C(imputation_method)'
model = ols(model_formula, data=results).fit()
anova_results = sm.stats.anova_lm(model, typ=3)

ss_total = model.ess + model.ssr
partial_r2 = pd.DataFrame({
    'Partial_R2': anova_results['sum_sq'] / ss_total
})
partial_r2 = partial_r2.sort_values('Partial_R2', ascending=False)
print(partial_r2)

model_formula = 'forecast_mse ~ imputed_rmse + imputed_mae + imputed_r2 + imputed_kl' + \
                '+ imputed_ks + imputed_w2 + imputed_sliced_w2'
model = ols(model_formula, data=results).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

ss_total = model.ess + model.ssr
partial_r2 = pd.DataFrame({
    'Partial_R2': anova_results['sum_sq'] / ss_total
})
partial_r2 = partial_r2.sort_values('Partial_R2', ascending=False)
print(partial_r2)

model_formula = 'forecast_mae ~ imputed_rmse + imputed_mae + imputed_r2 + imputed_kl' + \
                '+ imputed_ks + imputed_w2 + imputed_sliced_w2'
model = ols(model_formula, data=results).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

ss_total = model.ess + model.ssr
partial_r2 = pd.DataFrame({
    'Partial_R2': anova_results['sum_sq'] / ss_total
})
partial_r2 = partial_r2.sort_values('Partial_R2', ascending=False)
print(partial_r2)