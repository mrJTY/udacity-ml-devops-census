# Model Card

## Model Details

**Model Type:** Random Forest Classifier

**Developer:** Developed as part of Udacity ML DevOps Engineer Nanodegree program

**Model Version:** 1.0

**Model Date:** 2025-11-03

**Framework:** scikit-learn (sklearn.ensemble.RandomForestClassifier)

**Hyperparameters:**
- `n_estimators`: 50
- `max_depth`: 5
- `random_state`: 123

**License:** Educational use

**Model Architecture:**
The model uses an ensemble of 50 decision trees with limited depth (max_depth=5) to prevent overfitting. Features are preprocessed using OneHotEncoding for categorical variables and concatenated with continuous numerical features. The target variable (salary) is binarized using LabelBinarizer.

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
- Target variable: LabelBinarizer (<=50K → 0, >50K → 1)
- Continuous features: No scaling applied (Random Forest is scale-invariant)

## Evaluation Data

**Source:** Same dataset as training data (UCI Adult Census Income)

**Size:** 20% of total dataset (~6,512 samples)

**Split Method:** Random stratified split to maintain class distribution

**Preprocessing:** Same transformations as training data, using fitted encoder and label binarizer from training phase

## Metrics

**Overall Model Performance:**

The model is evaluated using standard binary classification metrics:

- **Precision:** Proportion of positive predictions that are actually positive (TP / (TP + FP))
- **Recall:** Proportion of actual positives that are correctly identified (TP / (TP + FN))
- **F1-Score (F-beta with β=1):** Harmonic mean of precision and recall

**Expected Performance Range:**
- Precision: 0.70 - 0.75
- Recall: 0.60 - 0.65
- F1-Score: 0.65 - 0.70

*(Note: Exact values vary due to random train-test split)*

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

3. **Class Imbalance:** Dataset is imbalanced (~75% <=50K, ~25% >50K), which affects model predictions

4. **Shallow Model:** Max depth of 5 limits model complexity, potentially underfitting complex patterns

5. **No Feature Engineering:** Uses raw features without domain-specific transformations or feature selection

6. **Missing Values:** Dataset contains missing values (denoted as "?") that require handling

7. **Demographic Bias:** Model exhibits performance disparities across protected attributes

**Recommendations for Users:**

- **Do *not* use for production decisions:** This model is for educational purposes only
- Perform bias audits - Always evaluate fairness metrics before considering any application
- Update with recent data - Retrain with contemporary data if real-world application is considered
- Consider feature exclusion - Remove protected attributes from training if fairness is critical
- Apply bias mitigation - Use techniques like reweighting, threshold optimization, or adversarial debiasing
- TODO: Use K-fold cross-validation - Current implementation uses single train-test split; K-fold would provide more robust evaluation
- Implement monitoring - If deployed, continuously monitor for performance degradation and bias drift
- Provide recourse - Ensure affected individuals can contest predictions and understand decision factors

**Future Improvements:**

- Experiment with regularization and hyperparameter tuning
- Implement cross-validation for more robust performance estimates
- Explore feature engineering (e.g., age groups, income brackets)
- Compare with other algorithms (Gradient Boosting, Neural Networks)
- Implement fairness constraints during training
- Add explainability tools (SHAP, LIME) for prediction interpretation

**Contact Information:**

For questions about this model or to report issues, please refer to the project repository or course materials.

---

**Last Updated:** 2025-11-03
**Model Card Version:** 1.0
