# 8. Visualisations
sns.histplot(merged['bat_landing_to_food'],kde=True,color='teal')
plt.title('Distribution of Bat Landing to Food'); plt.show()
sns.boxplot(data=merged,x='season',y='bat_landing_to_food'); plt.show()
sns.heatmap(merged.select_dtypes(float).corr(),annot=True,cmap='coolwarm'); plt.title('Correlation Heatmap'); plt.show()

# 9. Hypothesis testing
risk_rat=merged.loc[merged['rat_present']==1,'risk']; risk_norat=merged.loc[merged['rat_present']==0,'risk']
print(f"T-test Risk~Rat Presence p={stats.ttest_ind(risk_rat,risk_norat,nan_policy='omit').pvalue:.4f}")
# Chi-square for season vs risk
ct=pd.crosstab(merged['season'],merged['risk']); chi2,p,_,_=stats.chi2_contingency(ct)
print(f"Chi-square Season vs Risk p={p:.4f}")

#  10. Regression analyses

#Building Formula
logit_formula = 'risk ~ rat_present + seconds_after_rat_arrival + C(season)'
if 'hours_after_sunset' in merged.columns:
    logit_formula = 'risk ~ rat_present + seconds_after_rat_arrival + hours_after_sunset + C(season)'

ols_formula = 'bat_landing_to_food ~ rat_present + rat_minutes + C(season)'
if 'hours_after_sunset' in merged.columns:
    ols_formula = 'bat_landing_to_food ~ rat_present + rat_minutes + hours_after_sunset + C(season)'

# Fit logistic and linear models
logit = smf.logit(logit_formula, data=merged).fit(disp=False)
print("\nLogistic Regression Summary:\n", logit.summary())

ols = smf.ols(ols_formula, data=merged).fit()
print("\nOLS Regression Summary:\n", ols.summary())

# === Residual diagnostics & RMSE ===
plt.scatter(ols.fittedvalues, ols.resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# RMSE calculation
rmse = np.sqrt(mean_squared_error(ols.model.endog, ols.fittedvalues))
print(f"RMSE = {rmse:.3f}")

# === Multicollinearity (VIF) ===
X = ols.model.exog
vif = pd.DataFrame({
    'Variable': ols.model.exog_names,
    'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
})
print("\nVIF Table:\n", vif)
