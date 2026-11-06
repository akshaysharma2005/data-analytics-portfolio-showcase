# 🚀 Advanced Data Analytics & Machine Learning Portfolio

<p align="center">
  <img src="https://img.shields.io/badge/Author-M%20Akshay%20Sharma-indigo?style=for-the-badge&logo=github" alt="Author">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/R-4.0%2B-276DC3?style=for-the-badge&logo=r" alt="R">
  <img src="https://img.shields.io/badge/Interactive-Web_Platforms-emerald?style=for-the-badge&logo=html5" alt="Web">
  <img src="https://img.shields.io/badge/License-MIT-amber?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>An executive portfolio of 5 production-grade data analytics, machine learning, and statistical modeling platforms</strong> spanning real estate economics, quantitative finance, biometrics telemetry, counter-terrorism intelligence, and public health epidemiology.
</p>

---

## 📑 Portfolio Overview & Live GitHub Repositories

Each project in this portfolio is an end-to-end analytical study equipped with dedicated **Jupyter Notebooks**, **91 high-resolution production visualizations**, and a **custom browser-native interactive analytics web application** with live simulators:

| # | Project | Domain | Architecture / Models | Primary Result | Dedicated GitHub Repository |
|---|---------|--------|----------------------|----------------|----------------------------|
| 1 | **[Cracow Real Estate AI](#1-cracow-residential-real-estate-valuation-ml)** | Real Estate Economics | VotingRegressor Ensemble (MLP + GBR) | **RMSE: 116,478 PLN (MSLE: 0.03)** | [cracow-real-estate-pricing](https://github.com/akshaysharma2005/cracow-real-estate-pricing) |
| 2 | **[Polar Watch Telemetry](#2-polar-watch-biometric-telemetry--mixedlm-modeling)** | Sports Biometrics | Statsmodels MixedLM (REML) | **RMSE: 61.2 kcal (-23% vs OLS)** | [polar-watch-fitness-analysis](https://github.com/akshaysharma2005/polar-watch-fitness-analysis) |
| 3 | **[FX Algorithmic Risk](#3-fx-algorithmic-trading-performance--monte-carlo-risk)** | Quantitative Finance | 100K Monte Carlo + Geometric Loss PMF | **Profit Factor: 1.60 (Payoff: 5.22x)** | [fx-trading-analysis](https://github.com/akshaysharma2005/fx-trading-analysis) |
| 4 | **[Global Terrorism Matrix](#4-global-terrorism-database-intelligence-matrix)** | Geopolitical Security | R Tidyverse Spatial Aggregation | **180K+ Fatalities (ISIL, Taliban)** | [global-terrorism-eda](https://github.com/akshaysharma2005/global-terrorism-eda) |
| 5 | **[COVID-19 Epidemiology](#5-covid-19-pandemic-trajectories--world-bank-matrix)** | Public Health | Day T0 Alignment + Bivariate OLS | **35M+ Cases (Rural Corr: -0.46)** | [covid19-pandemic-analysis](https://github.com/akshaysharma2005/covid19-pandemic-analysis) |

---

## 🌐 Local Interactive Web Platforms

Each project folder includes a modern interactive web application featuring **Chart.js** charts, **dark/light modes**, **live prediction calculators**, and **searchable data tables with CSV exports**.

To preview all platforms locally in your browser, launch the local HTTP server from the root directory:

```bash
# Start local server
python3 -m http.server 8000
```

Then open your browser:
- **Master Showcase Hub:** [http://localhost:8000/index.html](http://localhost:8000/index.html)
- **Cracow Real Estate AI:** [http://localhost:8000/cracow-real-estate-pricing/index.html](http://localhost:8000/cracow-real-estate-pricing/index.html)
- **Polar Watch Telemetry:** [http://localhost:8000/polar-watch-fitness-analysis/index.html](http://localhost:8000/polar-watch-fitness-analysis/index.html)
- **FX Algorithmic Risk:** [http://localhost:8000/fx-trading-analysis/index.html](http://localhost:8000/fx-trading-analysis/index.html)
- **Global Terrorism Matrix:** [http://localhost:8000/global-terrorism-eda/index.html](http://localhost:8000/global-terrorism-eda/index.html)
- **COVID-19 Epidemiology:** [http://localhost:8000/covid19-pandemic-analysis/index.html](http://localhost:8000/covid19-pandemic-analysis/index.html)

*(Note: All web platforms are zero-dependency static HTML/JS files; they can also be opened directly via `file:///` in any modern web browser).*

---

## 1. Cracow Residential Real Estate Valuation ML

Predicting residential flat sale prices across 20 districts in Cracow, Poland using web-scraped listings.

### Architecture & Pipeline
- **Preprocessing:** Categorical one-hot encoding, missing value imputation via K-Nearest Neighbors (KNN), outlier mitigation with log-transforms.
- **Feature Engineering:** Ratio features (area per room, price per m²), floor elevation categories, and district frequency weights.
- **Model Ensemble:** VotingRegressor combining a Multi-Layer Perceptron (MLP) neural network and Gradient Boosting Regressor (GBR).
- **Interactive Simulator:** Real-time property valuation engine estimating market value, price/m², 95% confidence intervals, and district benchmark deviations.

### Results Snapshot
- Baseline Dummy Regressor (DMR): RMSE 222,475 PLN | MAE 161,219 PLN | MSLE 0.13
- Multi-Layer Perceptron (MLP): RMSE 120,493 PLN | MAE 79,646 PLN | MSLE 0.03
- Gradient Boosting (GBR): RMSE 119,237 PLN | MAE 76,182 PLN | MSLE 0.03
- **Voting Ensemble (Best): RMSE 116,478 PLN | MAE 74,993 PLN | MSLE 0.03**

---

## 2. Polar Watch Biometric Telemetry & MixedLM Modeling

Statistical analysis of wearable sensor telemetry from a Polar Vantage fitness watch across 283 workouts.

### Architecture & Diagnostics
- **Collinearity Pruning:** Evaluated Variance Inflation Factors (VIF); eliminated severely inflated mean heart rate indicators (VIF > 38) in favor of the 99th percentile peak heart rate (HR Q99, VIF = 2.14).
- **Model Formulation:** Statsmodels Restricted Maximum Likelihood (REML) Mixed Linear Model:
  $$\text{kiloCalories} = -494.503 + 6.432 \times \text{time} + 4.282 \times \text{HR}_{q99} + b_{\text{sport}}$$
- **Random Effects:** Inter-sport variance ($\sigma^2_{\text{sport}} = 2.810$) and random duration slopes ($\sigma^2_{\text{time}} = 0.423$).
- **Performance:** Reduced RMSE from 79.4 kcal (OLS) to **61.2 kcal (MixedLM)**, achieving a 23% error reduction.
- **Interactive Simulator:** Biometric workout energy expenditure calculator and circadian workout schedule analyzer.

---

## 3. FX Algorithmic Trading Performance & Monte Carlo Risk

Performance evaluation and risk analysis of 92 executed trades across currency pairs (EURUSD, USDCHF, GBPUSD) and equity CFDs (US500).

### Key Findings
- **Payoff Asymmetry:** Average Win of \$56.63 vs Average Loss of -\$10.84 yields an asymmetric payoff ratio of **5.22x**, driving a **1.60 Profit Factor** despite a 40.22% win rate.
- **Geometric Consecutive Losses:** Verified empirical losing streaks against the geometric PMF $P(L \ge k) = (0.5978)^k$. Maximum observed consecutive loss run was 8 trades, resulting in a contained -\$86.72 drawdown.
- **Monte Carlo Engine:** 100,000 resampled simulations establishing the 5th, 50th, and 95th percentile equity cones and a risk of ruin $< 0.05\%$.
- **Interactive Simulator:** In-browser 1,000-path stochastic Monte Carlo equity curve visualizer with adjustable risk parameters.

---

## 4. Global Terrorism Database Intelligence Matrix

Geospatial exploratory data analysis of 83,606 incidents from 2007 through 2017 using R Tidyverse.

### Key Insights
- **Lethality Scaling:** 180,332 recorded fatalities; peak casualty surge occurred in 2014 coinciding with ISIL territorial expansion in Iraq and Syria.
- **Top 5 Deadliest Factions:** ISIL (12,507 kills), Taliban (10,721 kills), Boko Haram (10,242 kills), TTP Pakistan (1,543 kills), and Al-Shabaab (1,138 kills).
- **Tactical Shifts:** Improvised Explosive Devices (IEDs) account for 53.4% of incidents, followed by armed assaults (26.2%). Private citizens and civilian infrastructure represented 38.2% of all targeted entities.
- **Interactive Platform:** Decadal time-lapse incident player, group tactical radar comparison, and searchable incident archive.

---

## 5. COVID-19 Pandemic Trajectories & World Bank Matrix

Cross-national epidemiological trajectory analysis combining Johns Hopkins CSSE surveillance with World Bank socioeconomic indicators across 180+ sovereign nations.

### Key Insights
- **Day T0 Outbreak Alignment:** Synchronized national trajectories from the date each country reached 100 confirmed cases, benchmarking infection velocity against theoretical doubling curves (doubling every 2, 5, and 10 days).
- **Socioeconomic Correlations:**
  - **Rural Population %:** Strongest negative correlation with confirmed infection rates ($r = -0.46, R^2 = 0.21$), indicating heightened transmission vulnerability in dense urban corridors.
  - **GDP per Capita & Life Expectancy:** Positively correlated with reported case fatality ($r = +0.34$ and $r = +0.36$), reflecting older demographic distributions and higher diagnostic testing capacity.
- **Interactive Platform:** Country trajectory visualizer with logarithmic/linear scaling and bivariate socioeconomic scatter regression studio.

---

## 🛠️ Technology Stack

- **Machine Learning & Modeling:** Scikit-Learn (`VotingRegressor`, `MLPRegressor`, `GradientBoostingRegressor`, `KNNImputer`), Statsmodels (`MixedLM`), Scipy (`linregress`, `geom`).
- **Data Manipulation:** Pandas, NumPy, R Tidyverse (`dplyr`, `ggplot2`, `tidyr`, `lubridate`).
- **Data Visualization:** Matplotlib, Seaborn, GGplot2, Chart.js v4, Plotly.
- **Web Applications:** Tailwind CSS, HTML5, Vanilla ES6+ JavaScript, Lucide Icons, KaTeX Math.

---

## 👤 Author

**M Akshay Sharma**
- GitHub: [@akshaysharma2005](https://github.com/akshaysharma2005)
- Computer Science & Engineering | VIT-AP University
- Full Stack Web Developer & Data Analytics Enthusiast
