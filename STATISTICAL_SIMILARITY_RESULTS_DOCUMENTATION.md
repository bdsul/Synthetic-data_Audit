# Statistical Similarity Test Results Documentation

## Overview

This document provides comprehensive documentation for the statistical similarity evaluation between real and synthetic data generated using four different SDV (Synthetic Data Vault) models. The evaluation compares statistical properties of real datasets with their synthetic counterparts to assess the quality and fidelity of synthetic data generation.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Methodology](#methodology)
3. [Datasets](#datasets)
4. [Synthetic Data Generation Models](#synthetic-data-generation-models)
5. [Statistical Measures](#statistical-measures)
6. [Results Interpretation](#results-interpretation)
7. [Key Findings](#key-findings)
8. [Results File Structure](#results-file-structure)
9. [How to Use the Results](#how-to-use-the-results)
10. [Limitations and Considerations](#limitations-and-considerations)

---

## Executive Summary

This study evaluates the statistical similarity between real and synthetic data across three datasets (Bank, Cancer, and Alzhimers) using four SDV models: CTGAN, CopulaGAN, Gaussian Copula, and TVAE. The evaluation focuses on four key statistical measures:

- **Mean**: Central tendency measure
- **Median**: Robust central tendency measure
- **Standard Deviation**: Measure of variability
- **Outliers**: Extreme values detected using IQR method

The results are stored in `statistical_similarity_results.csv` and provide detailed comparisons for each dataset-model combination.

---

## Methodology

### 1. Data Loading and Preprocessing

#### Bank Dataset
- **Source**: `bank-full.csv`
- **Preprocessing**: 
  - Sampled first 10,000 rows for computational efficiency
  - Maintained original structure with all features
- **Features**: 17 columns including age, job, marital status, balance, duration, etc.

#### Cancer Dataset
- **Source**: `Cancer.csv`
- **Preprocessing**:
  - Removed non-feature columns (id, Unnamed: 32)
  - Converted diagnosis column: M→1, B→0
- **Features**: 31 numeric features related to cancer cell characteristics

#### Alzhimers Dataset
- **Source**: `Alzhimers.xlsx`
- **Preprocessing**:
  - Removed identifier columns (Subject ID, MRI ID, Hand, M/F, Group)
  - Handled missing values using median imputation
- **Features**: 10 numeric features related to brain measurements

### 2. Synthetic Data Generation

For each dataset, synthetic data was generated using four SDV models:

1. **CTGAN** (Conditional Tabular GAN)
2. **CopulaGAN** (GAN-based with copula modeling)
3. **Gaussian Copula** (Statistical copula-based approach)
4. **TVAE** (Tabular Variational Autoencoder)

Each model was:
- Trained on the real dataset
- Used to generate synthetic samples equal to the size of the real dataset
- Evaluated for statistical similarity

### 3. Statistical Evaluation

For each numeric column in each dataset, the following statistics were calculated:

#### Mean
- **Formula**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Purpose**: Measures central tendency
- **Comparison**: Absolute difference and relative error percentage

#### Median
- **Definition**: Middle value when data is sorted
- **Purpose**: Robust measure of central tendency (less sensitive to outliers)
- **Comparison**: Absolute difference and relative error percentage

#### Standard Deviation
- **Formula**: $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$
- **Purpose**: Measures variability/spread of data
- **Comparison**: Absolute difference and relative error percentage

#### Outliers (IQR Method)
- **Method**: Interquartile Range (IQR) method
- **Formula**: 
  - Q1 = 25th percentile
  - Q3 = 75th percentile
  - IQR = Q3 - Q1
  - Lower bound = Q1 - 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR
  - Outliers: values < lower_bound or > upper_bound
- **Purpose**: Identifies extreme values that may indicate data quality issues
- **Comparison**: Count difference between real and synthetic data

### 4. Comparison Metrics

For each statistical measure, two types of comparisons were performed:

1. **Absolute Difference**: $|value_{real} - value_{synthetic}|$
2. **Relative Error**: $\frac{|value_{real} - value_{synthetic}|}{|value_{real}|} \times 100\%$

Relative error is particularly useful as it provides a normalized measure of difference, making it easier to compare across different scales.

---

## Datasets

### Bank Dataset
- **Size**: 10,000 samples (sampled from larger dataset)
- **Features**: 17 columns
- **Numeric Features**: age, balance, duration, campaign, pdays, previous, day
- **Characteristics**: Banking customer data with mixed numeric and categorical features

### Cancer Dataset
- **Size**: 569 samples
- **Features**: 31 numeric columns
- **Characteristics**: Medical data with measurements of cancer cell characteristics
- **Key Features**: radius, texture, perimeter, area, smoothness, compactness, concavity, etc.

### Alzhimers Dataset
- **Size**: 373 samples
- **Features**: 10 numeric columns
- **Characteristics**: Medical imaging data related to Alzheimer's disease
- **Key Features**: Visit, MR Delay, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF

---

## Synthetic Data Generation Models

### 1. CTGAN (Conditional Tabular GAN)
- **Type**: Generative Adversarial Network
- **Strengths**: 
  - Can capture complex non-linear relationships
  - Good for high-dimensional data
- **Weaknesses**: 
  - Requires more training time
  - May be unstable during training

### 2. CopulaGAN
- **Type**: GAN with copula modeling
- **Strengths**:
  - Combines GAN capabilities with statistical copula theory
  - Better handling of mixed data types
- **Weaknesses**:
  - More complex architecture
  - May require more hyperparameter tuning

### 3. Gaussian Copula
- **Type**: Statistical/Probabilistic model
- **Strengths**:
  - Fast training and generation
  - Interpretable model
  - Good baseline performance
- **Weaknesses**:
  - Assumes Gaussian relationships
  - May not capture complex non-linear patterns

### 4. TVAE (Tabular Variational Autoencoder)
- **Type**: Variational Autoencoder
- **Strengths**:
  - Stable training
  - Good reconstruction capabilities
  - Can learn meaningful latent representations
- **Weaknesses**:
  - May produce smoother distributions
  - Less sharp than GAN-based models

---

## Statistical Measures

### Mean
The arithmetic mean represents the average value of a feature. A low relative error in mean indicates that the synthetic data maintains the same central tendency as the real data.

**Interpretation**:
- **< 5% error**: Excellent match
- **5-15% error**: Good match
- **15-30% error**: Moderate match
- **> 30% error**: Poor match

### Median
The median is the middle value and is less sensitive to outliers than the mean. A good median match indicates that the synthetic data preserves the distribution's center point.

**Interpretation**:
- **< 5% error**: Excellent match
- **5-15% error**: Good match
- **15-30% error**: Moderate match
- **> 30% error**: Poor match

### Standard Deviation
Standard deviation measures the spread or variability of the data. A good match indicates that the synthetic data has similar variability to the real data.

**Interpretation**:
- **< 10% error**: Excellent match
- **10-25% error**: Good match
- **25-50% error**: Moderate match
- **> 50% error**: Poor match

### Outliers
Outliers are extreme values that fall outside the normal range. The IQR method identifies values beyond 1.5 × IQR from the quartiles.

**Interpretation**:
- **Similar count**: Synthetic data captures extreme values well
- **Large difference**: Model may be smoothing out extremes or generating unrealistic values

---

## Results Interpretation

### Reading the Results CSV

The `statistical_similarity_results.csv` file contains the following columns:

1. **Dataset**: Name of the dataset (Bank, Cancer, or Alzhimers)
2. **Model**: SDV model used (CTGAN, CopulaGAN, GaussianCopula, or TVAE)
3. **Column**: Feature name being evaluated
4. **Mean_Real**: Mean value from real data
5. **Mean_Synthetic**: Mean value from synthetic data
6. **Mean_Error_%**: Relative error percentage for mean
7. **Median_Real**: Median value from real data
8. **Median_Synthetic**: Median value from synthetic data
9. **Median_Error_%**: Relative error percentage for median
10. **Std_Real**: Standard deviation from real data
11. **Std_Synthetic**: Standard deviation from synthetic data
12. **Std_Error_%**: Relative error percentage for standard deviation
13. **Outliers_Real**: Number of outliers in real data
14. **Outliers_Synthetic**: Number of outliers in synthetic data
15. **Outliers_Diff**: Absolute difference in outlier counts

### Example Interpretation

Consider a row from the results:
```
Dataset: Cancer
Model: CTGAN
Column: area_mean
Mean_Real: 654.89
Mean_Synthetic: 1052.97
Mean_Error_%: 60.79
```

This indicates:
- The real data has an average area_mean of 654.89
- The synthetic data has an average area_mean of 1052.97
- There is a 60.79% relative error, indicating a **poor match** for this feature
- The model is overestimating the mean area by approximately 60%

---

## Key Findings

### Overall Performance

Based on the results analysis:

1. **Gaussian Copula** typically shows:
   - Fastest generation time
   - Good performance on simpler datasets
   - May struggle with complex non-linear relationships

2. **CTGAN** typically shows:
   - Better capture of complex patterns
   - Good performance on high-dimensional data
   - May have higher variance in results

3. **CopulaGAN** typically shows:
   - Balanced performance
   - Good handling of mixed data types
   - Moderate computational requirements

4. **TVAE** typically shows:
   - Stable and consistent results
   - Good reconstruction quality
   - May produce smoother distributions

### Dataset-Specific Observations

#### Bank Dataset
- Contains mixed numeric and categorical features
- Some models may struggle with categorical encoding
- Balance and duration features show varying performance across models

#### Cancer Dataset
- High-dimensional dataset (31 features)
- Some features (like area_mean) show high error rates
- Models may struggle with capturing extreme values

#### Alzhimers Dataset
- Smaller dataset (373 samples)
- May benefit from simpler models
- Missing value imputation may affect results

### Common Patterns

1. **Mean vs Median**: 
   - Median errors are often lower than mean errors, indicating that models preserve central values better than accounting for outliers

2. **Standard Deviation**:
   - Often shows higher error rates, suggesting models may struggle to capture the full variability of real data

3. **Outliers**:
   - Significant differences in outlier counts suggest models may be smoothing distributions or failing to capture extreme values

---

## Results File Structure

### CSV Format

The results file is a standard CSV with the following structure:

```
Dataset,Model,Column,Mean_Real,Mean_Synthetic,Mean_Error_%,...
```

### Filtering Results

You can filter the results by:

1. **Dataset**: Filter to specific dataset (Bank, Cancer, Alzhimers)
2. **Model**: Filter to specific model (CTGAN, CopulaGAN, GaussianCopula, TVAE)
3. **Error Threshold**: Filter columns with error above/below certain thresholds

### Example Queries

**Find all columns with mean error > 30%:**
```python
import pandas as pd
results = pd.read_csv('statistical_similarity_results.csv')
high_error = results[results['Mean_Error_%'] > 30]
```

**Compare models for a specific dataset:**
```python
cancer_results = results[results['Dataset'] == 'Cancer']
model_comparison = cancer_results.groupby('Model')['Mean_Error_%'].mean()
```

**Find best performing model for each dataset:**
```python
best_models = results.groupby(['Dataset', 'Model'])['Mean_Error_%'].mean().reset_index()
best_models = best_models.loc[best_models.groupby('Dataset')['Mean_Error_%'].idxmin()]
```

---

## How to Use the Results

### 1. Model Selection

Use the results to select the best model for your specific use case:

- **If accuracy is critical**: Choose model with lowest average error rates
- **If speed is critical**: Consider Gaussian Copula (fastest)
- **If handling complex relationships**: Consider CTGAN or CopulaGAN
- **If stability is important**: Consider TVAE

### 2. Feature Analysis

Identify problematic features:
- Features with consistently high error rates across all models may indicate:
  - Data quality issues in original data
  - Features that are difficult to model
  - Need for feature engineering

### 3. Quality Assessment

Use error thresholds to assess overall quality:
- **Excellent**: < 5% mean error, < 10% std error
- **Good**: 5-15% mean error, 10-25% std error
- **Acceptable**: 15-30% mean error, 25-50% std error
- **Poor**: > 30% mean error, > 50% std error

### 4. Model Improvement

Use results to guide model improvement:
- High mean errors: Adjust model hyperparameters or try different architectures
- High std errors: Models may need more training or different loss functions
- Outlier mismatches: Consider outlier-aware training or post-processing

---

## Limitations and Considerations

### 1. Statistical Measures Only

This evaluation focuses on statistical properties but does not assess:
- Privacy preservation
- Utility for downstream tasks
- Correlation preservation
- Temporal patterns (if applicable)

### 2. IQR Outlier Detection

The IQR method is one approach to outlier detection:
- May not capture all types of outliers
- Assumes normal-like distribution
- 1.5 × IQR is a common but arbitrary threshold

### 3. Relative Error Interpretation

Relative errors can be misleading when:
- Real values are close to zero (division by small numbers)
- Values have different scales (percentage may not be meaningful)
- Consider absolute differences for very small values

### 4. Dataset Size

Smaller datasets (like Alzhimers with 373 samples) may:
- Show higher variance in results
- Benefit from simpler models
- Require careful validation

### 5. Missing Values

Missing value handling (median imputation for Alzhimers) may:
- Affect the distribution of synthetic data
- Introduce bias in the results
- Consider multiple imputation strategies

### 6. Model Training

Results depend on:
- Training hyperparameters (not specified in this evaluation)
- Random seeds (may affect reproducibility)
- Training duration (may need more epochs)

---

## Recommendations

### For Practitioners

1. **Start with Gaussian Copula** for a quick baseline
2. **Use CTGAN or CopulaGAN** for complex datasets
3. **Validate on downstream tasks** beyond statistical similarity
4. **Monitor outlier preservation** if extreme values are important
5. **Consider ensemble approaches** combining multiple models

### For Researchers

1. **Extend evaluation** to include privacy metrics
2. **Assess utility** on machine learning tasks
3. **Compare with other synthetic data generation methods**
4. **Investigate feature-specific performance** patterns
5. **Study the relationship** between statistical similarity and utility

---

## Conclusion

This statistical similarity evaluation provides a comprehensive assessment of synthetic data quality across multiple datasets and models. The results can guide:

- **Model selection** for specific use cases
- **Quality assessment** of synthetic data
- **Identification of problematic features**
- **Model improvement** strategies

However, statistical similarity is just one aspect of synthetic data quality. Consider complementing this analysis with:

- Privacy evaluation
- Utility testing on downstream tasks
- Domain expert review
- Correlation and dependency analysis

---

## Contact and Support

For questions about these results or the evaluation methodology, please refer to:
- The original notebook: `statistical_similarity_test.ipynb`
- SDV documentation: https://sdv.dev/
- Results file: `statistical_similarity_results.csv`

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Evaluation Date**: See notebook execution timestamp

