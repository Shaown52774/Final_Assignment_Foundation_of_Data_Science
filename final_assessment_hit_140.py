# 4. Feature engineering
cols = merged.columns

# 1. Rat presence indicator
if all(c in cols for c in ['start_time','rat_period_start','rat_period_end']):
    merged['rat_present'] = np.where(
        (merged['start_time'] >= merged['rat_period_start']) &
        (merged['start_time'] <= merged['rat_period_end']), 1, 0)
else:
    merged['rat_present'] = 0

# 2. Time window since rat arrival
if 'seconds_after_rat_arrival' in cols:
    merged['recent_rat_window'] = pd.cut(
        merged['seconds_after_rat_arrival'],
        bins=[-np.inf, 30, 120, np.inf],
        labels=['≤30s', '30–120s', '>120s']
    )
else:
    merged['recent_rat_window'] = 'Unknown'

# 3. Hours after sunset (if available)
if 'hours_after_sunset' in cols:
    merged['after_sunset_bin'] = pd.cut(
        merged['hours_after_sunset'],
        bins=[-np.inf, 2, 4, np.inf],
        labels=['Early', 'Mid', 'Late']
    )
else:
    merged['after_sunset_bin'] = 'Unknown'

#5. Missing-value handling
print("Missing values: ", merged.isna().sum())
merged.fillna(merged.median(numeric_only=True), inplace=True)

# 6. Descriptive stats and dispersion
# Restrict calculations to numeric columns only
num = merged.select_dtypes(include=['number'])
desc = num.describe().T

# Add dispersion and shape measures
desc['skew'] = num.skew()
desc['kurtosis'] = num.kurtosis()
desc['cv'] = desc['std'] / desc['mean']

print("\nDescriptive Summary with CV:\n",
      desc[['mean', 'std', 'cv', 'skew', 'kurtosis']].round(3))

# 7. Normality checks
plt.figure(figsize=(6,4)); stats.probplot(merged['bat_landing_to_food'].dropna(), dist="norm", plot=plt)
plt.title('Q–Q Plot for Bat Landing to Food'); plt.show()
stat,p=stats.shapiro(merged['bat_landing_to_food'].dropna())
print(f"Shapiro–Wilk normality test p={p:.4f}")
merged['z_score_food']=(merged['bat_landing_to_food']-merged['bat_landing_to_food'].mean())/merged['bat_landing_to_food'].std()

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
