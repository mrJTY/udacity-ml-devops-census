# Classification Metrics

## TPR (True Positive Rate) - Aka "Sensitivity" or "Recall"
- Formula: TP / (TP + FN)
- Meaning: Of all actual positive cases, what percentage did the model correctly identify?
- Example: If 100 people actually earn >50K, and the model correctly identifies 45 of them, TPR = 0.45
- Why it matters: Low TPR for a group means the model is missing opportunities for that group

## FPR (False Positive Rate)
- Formula: FP / (FP + TN)
- Meaning: Of all actual negative cases, what percentage did the model incorrectly predict as positive?
- Example: If 1000 people earn ≤50K, and the model incorrectly predicts 36 earn >50K, FPR = 0.036
- Why it matters: High FPR means the model is making false promises to a group

## FDR (False Discovery Rate)
- Formula: FP / (FP + TP) = 1 - Precision
- Meaning: Of all positive predictions, what percentage is wrong?
- Example: Model predicts 100 people earn >50K, but 18 actually don't, FDR = 0.18
- Why it matters: High FDR means when the model says "yes" for this group, it's often wrong

## FOR (False Omission Rate)
- Formula: FN / (FN + TN)
- Meaning: Of all negative predictions, what percentage is wrong?
- Example: Model predicts 1000 people earn ≤50K, but 90 actually earn >50K, FOR = 0.09
- Why it matters: High FOR means the model is incorrectly denying opportunities to this group

##  Precision (PPV - Positive Predictive Value)
- Formula: TP / (TP + FP) = 1 - FDR
- Meaning: Of all positive predictions, what percentage is correct?
- Why it matters: High precision means you can trust the model's positive predictions

##  NPV (Negative Predictive Value)
- Formula: TN / (TN + FN) = 1 - FOR
- Meaning: Of all negative predictions, what percentage is correct?
- Why it matters: High NPV means you can trust the model's negative predictions

#  Disparity Metrics

## Disparity Ratio
- Formula: Metric for Group A / Metric for Reference Group
- Meaning: How much does a metric differ between groups?
- Example: If White TPR = 0.45 and Black TPR = 0.32, disparity = 0.32/0.45 = 0.71
- Interpretation:
- 1.0 = Perfect parity (no disparity)
- < 1.0 = Group is disadvantaged compared to reference
- > 1.0 = Group is advantaged compared to reference
- Common threshold: 0.8-1.25 is often considered "fair" (80% rule)

## PPR (Predicted Positive Ratio)
- Meaning: What proportion of a group received positive predictions?
- Why it matters: Shows if the model treats groups differently in terms of positive prediction rates

## PPR Disparity
- Meaning: Compares positive prediction rates across groups
- Example: If 2.3% of Group A gets positive predictions vs 44% of Reference Group, PPR disparity = 0.052
- Why it matters: Large differences suggest the model systematically favors/disfavors certain groups.

# Examples
- Female TPR = 0.19 vs Male TPR = 0.49
    - The model correctly identifies only 19% of high-earning females vs 49% of high-earning males
    - TPR disparity = 0.39 - females are significantly disadvantaged
- White TPR = 0.45 vs Black TPR = 0.32
    - Model misses more high earners in the Black group
    - This could perpetuate existing inequalities
- FPR is very low for females (0.001) vs males (0.054)
    - Model rarely predicts females will earn >50K (even when wrong)
    - This conservative approach means fewer false promises but also fewer opportunities for females to come up with >50k