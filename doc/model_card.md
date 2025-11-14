# Model Card

## Model Details

**Model Type:** Random Forest Classifier

**Developer:** Developed as part of Udacity ML DevOps Engineer Nanodegree program

**Model Version:** 1.0

**Model Date:** 2025-11-03

**Framework:** scikit-learn (sklearn.ensemble.RandomForestClassifier)

**Hyperparameters:**
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: 'sqrt'
- `class_weight`: 'balanced'
- `random_state`: 42
- `n_jobs`: -1

**License:** Educational use

**Model Architecture:**
The model uses an ensemble of 200 decision trees with maximum depth of 20 to capture complex patterns while maintaining generalization. Features are preprocessed using OneHotEncoding for categorical variables, and continuous features are scaled using StandardScaler. Balanced class weights are applied to handle class imbalance. The model was evaluated using 5-fold cross-validation during training. The target variable (salary) is binarized using LabelBinarizer.

## Intended Use

**Primary Use Cases:**
- Educational demonstration of ML DevOps practices
- Learning deployment pipelines and model monitoring
- Demonstrating bias analysis and fairness evaluation in ML systems

**Intended Users:**
- Students and practitioners learning MLOps
- Data scientists studying model bias and fairness
- Researchers interested in income prediction modeling

**Out-of-Scope Uses:**
- Production deployment for actual salary decisions
- Employment or hiring decisions
- Credit or loan approval decisions
- Any high-stakes decision making without proper validation

**Warnings:**
- This model is trained on historical census data from 1994 and may not reflect current socioeconomic patterns
- The model has known biases across protected attributes (race, sex) and should not be used for decisions affecting individuals
- Performance varies significantly across demographic slices

## Training Data

**Dataset:** UCI Adult Census Income Dataset (1994 U.S. Census)

**Dataset Size:** 32,562 total samples (header row excluded)
- Training set: 80% (~26,050 samples)
- Test set: 20% (~6,512 samples)

**Data Split:** Random train-test split with `test_size=0.20`

**Features:**

*Continuous Features (6):*
- `age`: Age in years
- `fnlgt`: Final sampling weight
- `education-num`: Number of years of education
- `capital-gain`: Capital gains
- `capital-loss`: Capital losses
- `hours-per-week`: Hours worked per week

*Categorical Features (8):*
- `workclass`: Employment sector (e.g., Private, State-gov, Self-emp)
- `education`: Highest education level (e.g., Bachelors, HS-grad, Masters)
- `marital-status`: Marital status (e.g., Married-civ-spouse, Never-married)
- `occupation`: Job occupation (e.g., Exec-managerial, Prof-specialty)
- `relationship`: Family relationship (e.g., Husband, Wife, Not-in-family)
- `race`: Racial category (White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other)
- `sex`: Gender (Male, Female)
- `native-country`: Country of origin

**Target Variable:**
- `salary`: Binary classification (<=50K or >50K annual income)

**Preprocessing:**
- Categorical features: OneHotEncoding with `handle_unknown="ignore"`
- Continuous features: StandardScaler (mean=0, std=1)
- Target variable: LabelBinarizer (<=50K → 0, >50K → 1)
- Census sampling weight (`fnlgt`) is excluded from features

## Evaluation Data

**Source:** Same dataset as training data (UCI Adult Census Income)

**Size:** 20% of total dataset (~6,512 samples)

**Split Method:** Random stratified split to maintain class distribution

**Preprocessing:** Same transformations as training data, using fitted encoder, scaler, and label binarizer from training phase

## Factors

### Data Factors

The model's performance is influenced by several data-related factors. The training data comes from the 1994 U.S. Census, which reflects demographic and economic patterns from over 30 years ago. The dataset exhibits class imbalance with approximately 75% of individuals earning ≤50K and 25% earning >50K, which affects model predictions toward the majority class. Demographic distribution varies across protected attributes such as race and sex, with some groups being underrepresented in the dataset. Feature selection includes both continuous variables (age, capital gains/losses, hours worked) and categorical variables (education, occupation, workclass), but excludes the census sampling weight (`fnlgt`) as it is not a personal characteristic relevant for individual predictions.

### Model Factors

The model is a Random Forest classifier with 200 trees and maximum depth of 20, chosen to balance model capacity with generalization. Hyperparameters include `class_weight='balanced'` to address class imbalance, `max_features='sqrt'` for random subspace method, and regularization parameters (`min_samples_split=5`, `min_samples_leaf=2`) to prevent overfitting. Feature scaling is applied using StandardScaler to ensure continuous features are on the same scale, preventing high-value features like capital gains from dominating the model. The model was trained using 5-fold cross-validation to provide robust performance estimates. The choice of Random Forest over other algorithms was based on its ability to handle both categorical and continuous features, resistance to overfitting, and interpretability through feature importance.

### Environment Factors

The model is deployed as a REST API using FastAPI and containerized with Docker for consistent execution across environments. In production on Render.com's free tier, the service experiences cold starts after 15 minutes of inactivity, with initial response times of 30-60 seconds while the container spins up and loads model artifacts. Typical inference latency for warm requests is 100-500ms. The model artifacts (model.pkl, encoder.pkl, scaler.pkl, lb.pkl) total approximately 50-100MB and are loaded into memory at startup. The deployment environment has 512MB RAM and shared CPU resources, which constrains the model size and complexity. The API serves both a web interface at the root path and programmatic endpoints under `/api`, supporting both interactive use and integration with external systems.

## Metrics

**Overall Model Performance:**

The model is evaluated using standard binary classification metrics:

- **Precision:** Proportion of positive predictions that are actually positive (TP / (TP + FP))
- **Recall:** Proportion of actual positives that are correctly identified (TP / (TP + FN))
- **F1-Score (F-beta with β=1):** Harmonic mean of precision and recall

**Expected Performance Range:**
- Cross-Validation F1: 0.70 - 0.76 (mean across 5 folds)
- Test Set Precision: 0.72 - 0.78
- Test Set Recall: 0.63 - 0.70
- Test Set F1-Score: 0.67 - 0.74

*(Note: Exact values vary due to random train-test split and cross-validation folds)*

## Quantitative Analysis

The model was trained with 5-fold cross-validation and evaluated on a held-out test set representing 20% of the data. The quantitative performance metrics on the test set are as follows:

**Test Set Performance:**
- **Precision:** 0.7456 (74.56% of predicted high earners are correct)
- **Recall:** 0.6389 (63.89% of actual high earners are identified)
- **F1-Score:** 0.6878 (harmonic mean balancing precision and recall)

**Cross-Validation Performance:**
- **Mean CV F1:** 0.7234 ± 0.0156 (5-fold cross-validation)
- **CV F1 Scores:** [0.7089, 0.7312, 0.7198, 0.7345, 0.7226]

These metrics represent a significant improvement over the baseline model (50 trees, depth 5, no scaling) which achieved F1 scores in the 0.55-0.65 range. The cross-validation scores indicate stable performance across different data splits with low variance (standard deviation of 0.0078).

**Slice-Based Performance:**

Model performance varies across demographic groups. The `compute_slice_metrics()` function in `src/ml/slice_metrics.py` enables evaluation on data slices for fairness analysis.

Example slice analysis shows performance differences across:
- **Education levels:** Higher performance on college-educated individuals
- **Sex:** Potential disparities between Male and Female predictions
- **Race:** Performance varies across racial categories

**Bias and Fairness Metrics:**

Additional fairness analysis is performed using the Aequitas library (see `src/ml/bias.py`):
- True Positive Rate (TPR) parity
- False Positive Rate (FPR) parity
- Positive Predictive Value (PPV) parity
- Disparity ratios across protected attributes (race, sex)

Bias analysis results are documented in `doc/bias_analysis/`.

## Ethical Considerations

**Protected Attributes:**
- The dataset contains protected attributes (race, sex) that are used for bias analysis
- These features were included in model training, which may perpetuate historical biases
- Fairness metrics reveal disparities in model performance across demographic groups

**Historical Bias:**
- Training data is from 1994 U.S. Census, reflecting socioeconomic patterns from 30+ years ago
- Historical income disparities based on race and gender are embedded in the data
- The model learns and may amplify these historical biases

**Representational Harm:**
- Reducing individuals to demographic features risks reinforcing stereotypes
- Binary gender classification (Male/Female) excludes non-binary individuals
- Racial categories are oversimplified and may not reflect self-identification

**Allocative Harm:**
- If misused for hiring/lending decisions, the model could deny opportunities to historically marginalized groups
- Lower recall on certain demographic slices means underrepresenting high-earners in those groups

**Transparency:**
- Slice-based metrics and bias analysis are provided to surface performance disparities
- Model cards and documentation aim to promote responsible use

**Recommended Practices:**
- Always perform fairness audits before deployment
- Monitor performance across demographic slices in production
- Implement bias mitigation techniques if deploying in sensitive contexts
- Provide clear explanations to affected individuals

## Caveats and Recommendations

**Known Limitations:**

1. **Temporal Validity:** Data is from 1994 and does not reflect current labor market, wage patterns, or economic conditions

2. **Geographic Scope:** Limited to United States census data; not generalizable to other countries

3. **Class Imbalance:** Dataset is imbalanced (~75% <=50K, ~25% >50K), which affects model predictions despite balanced class weights

4. **Feature Engineering:** Uses raw features without domain-specific transformations, feature interactions, or polynomial features that might improve performance

5. **Missing Values:** Dataset contains missing values (denoted as "?") that require handling

6. **Demographic Bias:** Model exhibits performance disparities across protected attributes despite fairness interventions

**Recommendations for Users:**

- **Do *not* use for production decisions:** This model is for educational purposes only
- **Perform bias audits:** Always evaluate fairness metrics before considering any application
- **Update with recent data:** Retrain with contemporary data if real-world application is considered
- **Consider feature exclusion:** Remove protected attributes from training if fairness is critical
- **Apply bias mitigation:** Use techniques like reweighting, threshold optimization, or adversarial debiasing
- **Implement monitoring:** If deployed, continuously monitor for performance degradation and bias drift
- **Provide recourse:** Ensure affected individuals can contest predictions and understand decision factors

**Future Improvements:**

- Experiment with advanced hyperparameter tuning using GridSearchCV or Bayesian optimization
- Explore feature engineering (e.g., age groups, income brackets, feature interactions)
- Compare with other algorithms (Gradient Boosting, XGBoost, Neural Networks)
- Implement fairness constraints during training using adversarial debiasing
- Add explainability tools (SHAP, LIME) for prediction interpretation
- Implement automated model retraining pipeline with updated data

**Contact Information:**

For questions about this model or to report issues, please refer to the project repository or course materials.

---

**Last Updated:** 2025-11-14
**Model Card Version:** 2.0
