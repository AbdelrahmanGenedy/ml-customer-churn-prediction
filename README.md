# Customer Churn Prediction - ML Pipeline

A comprehensive data science project demonstrating end-to-end machine learning workflow for predicting customer churn in the telecommunications industry.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Business Impact](#business-impact)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

Customer churn represents a critical challenge for telecom companies, where acquiring new customers costs 5-7x more than retaining existing ones. This project addresses the need for proactive identification of at-risk customers, enabling targeted retention strategies before cancellation occurs.

### Business Problem
- **Challenge**: High customer churn rate impacting revenue
- **Solution**: Predictive ML model to identify at-risk customers
- **Impact**: Potential $47M annual revenue savings

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | Random Forest |
| **AUC Score** | 0.78 |
| **Recall** | 70% (catches 7 out of 10 churners) |
| **Precision** | 64% |
| **Dataset Size** | 5,000 customers |
| **Features** | 18 engineered features |

### Top Churn Predictors
1. **Tenure** (28% importance) - New customers (<12 months) show 3x higher churn
2. **Contract Type** (19% importance) - Month-to-month = 3x risk vs. 2-year contracts
3. **Monthly Charges** (15% importance) - High bills correlate with dissatisfaction
4. **Service Calls** (12% importance) - 3+ calls indicate unresolved issues

## 🛠 Technologies Used

- **Language**: Python 3.8+
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Algorithms**: 
  - Logistic Regression (baseline)
  - Random Forest (best performer)
  - Gradient Boosting

## 💻 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### Run the Complete Pipeline
```bash
python customer_churn_prediction.py
```

### Output Files
The script generates the following outputs:
- `eda_visualizations.png` - Exploratory data analysis charts
- `feature_importance.png` - Top predictive features visualization
- `roc_curves.png` - Model performance comparison
- `model_comparison.csv` - Summary of model metrics

### Jupyter Notebook (Optional)
For interactive exploration:
```bash
jupyter notebook customer_churn_analysis.ipynb
```

## 📁 Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.py    # Main script
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── LICENSE                         # License file
│
├── data/                          # Data directory (generated)
│   └── synthetic_customer_data.csv
│
├── outputs/                       # Generated outputs
│   ├── eda_visualizations.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── model_comparison.csv
│
├── models/                        # Saved models (optional)
│   └── random_forest_model.pkl
│
└── notebooks/                     # Jupyter notebooks
    └── customer_churn_analysis.ipynb
```

## 🔬 Methodology

### 1. Data Generation
- Created realistic synthetic dataset with 5,000 customer records
- 15+ features including tenure, contract type, monthly charges, service usage
- Target variable engineered based on industry churn patterns

### 2. Exploratory Data Analysis
- Analyzed churn distribution across different customer segments
- Identified correlations between features and churn
- Visualized key trends and patterns

### 3. Feature Engineering
- **New Features**:
  - `charges_per_month`: Average monthly spending rate
  - `has_internet`: Binary internet service indicator
  - `has_premium_services`: Aggregated premium features flag
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### 4. Model Development
Trained and compared three classification models:
- **Logistic Regression**: Baseline interpretable model
- **Random Forest**: Ensemble method for non-linear patterns
- **Gradient Boosting**: Sequential learning approach

### 5. Evaluation
- 80/20 train-test split with stratification
- **Primary metric**: AUC-ROC (0.78)
- **Secondary metrics**: Precision, Recall, F1-Score
- Feature importance analysis for interpretability

## 💼 Business Impact

### Retention Strategies
1. **Contract Upgrades**: Incentivize month-to-month → annual plans (20% discount)
2. **Premium Bundling**: Tech support + online security at 30% off for at-risk customers
3. **Loyalty Pricing**: Personalized offers for high-paying customers

### Proactive Outreach
- Monthly customer scoring with automated alerts (>70% churn probability)
- Specialized onboarding program for first 12 months
- Proactive support after 3rd service call

### ROI Estimate
- **Customer LTV**: $1,500
- **Monthly at-risk customers**: 12,500 (25% of 50K base)
- **Retention rate improvement**: 30% of flagged customers
- **Annual revenue impact**: **$47.25M saved**

## 🚀 Future Enhancements

- [ ] **Model Deployment**: Containerize with Docker and deploy to AWS/Azure
- [ ] **Real-time API**: Flask/FastAPI endpoint for live predictions
- [ ] **Dashboard**: Tableau/Power BI executive dashboard
- [ ] **A/B Testing**: Validate retention campaign effectiveness
- [ ] **Advanced Models**: XGBoost, Neural Networks, SHAP explainability
- [ ] **Customer Segmentation**: K-means clustering for personalized strategies
- [ ] **Time Series**: Incorporate temporal patterns and seasonality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Abdelrahman Genedy**
- LinkedIn: Abdelrahman Genedy
- GitHub: [@AbdelrahmanGenedy](https://github.com/AbdelrahmanGenedy)

## 📧 Contact

Questions or feedback? Feel free to reach out!
- Email: abdelrahmangenedy01@gmail.com

---

⭐ If you found this project helpful, please consider giving it a star!

## 📚 References

- Scikit-learn Documentation: https://scikit-learn.org/
- Customer Churn Analysis Best Practices
- Telecom Industry Benchmarks

---

**Last Updated**: February 2026
