import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

results = pd.read_csv('./results.csv')

def calculate_and_plot_partial_r2(formula, data, typ, title):
    model = ols(formula, data=data).fit()
    anova_results = sm.stats.anova_lm(model, typ=typ)
    
    ss_total = model.ess + model.ssr
    partial_r2 = pd.DataFrame({
        'Partial_R2': anova_results['sum_sq'] / ss_total
    })
    
    # Filter out Residual and Intercept
    partial_r2 = partial_r2.drop(['Residual', 'Intercept'], errors='ignore')
    partial_r2 = partial_r2.sort_values('Partial_R2', ascending=True)  # Ascending for better visualization
    
    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=partial_r2['Partial_R2'], y=partial_r2.index, palette='viridis', hue=partial_r2.index, legend=False)
    plt.title(title)
    plt.xlabel('Partial R²')
    plt.ylabel('Factors')
    plt.tight_layout()
    
    # Add values to end of bars
    # Add values to end of bars for most bars, but put max value inside the bar
    max_idx = partial_r2['Partial_R2'].idxmax()
    for i, (idx, v) in enumerate(zip(partial_r2.index, partial_r2['Partial_R2'])):
        if idx == max_idx:
            # For max value, place text inside the bar with contrasting color
            ax.text(v - 0.001, i, f"{v:.4f}", va='center', ha='right')
        else:
            # For other values, keep them at the end of bars
            ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    
    plt.savefig(f'./visualization/{title.replace(" ", "_")}.png', dpi=300)
    plt.close()
    
    return partial_r2

# Generate plots for each model
mse_factors = calculate_and_plot_partial_r2(
    'forecast_mse ~ missing_rate * C(missing_type) * completeness_rate * C(imputation_method)',
    results, 3, 'Partial R² for MSE by Experimental Factors'
)

mae_factors = calculate_and_plot_partial_r2(
    'forecast_mae ~ missing_rate * C(missing_type) * completeness_rate * C(imputation_method)',
    results, 3, 'Partial R² for MAE by Experimental Factors'
)

mse_metrics = calculate_and_plot_partial_r2(
    'forecast_mse ~ imputed_rmse + imputed_mae + imputed_r2 + imputed_kl + imputed_ks + imputed_w2 + imputed_sliced_w2',
    results, 2, 'Partial R² for MSE by Imputation Quality Metrics'
)

mae_metrics = calculate_and_plot_partial_r2(
    'forecast_mae ~ imputed_rmse + imputed_mae + imputed_r2 + imputed_kl + imputed_ks + imputed_w2 + imputed_sliced_w2',
    results, 2, 'Partial R² for MAE by Imputation Quality Metrics'
)
