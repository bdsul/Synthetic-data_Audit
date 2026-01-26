# Statistical Similarity Results - Quick Reference Guide

## Overview
This is a quick reference guide for interpreting the statistical similarity test results. For detailed documentation, see `STATISTICAL_SIMILARITY_RESULTS_DOCUMENTATION.md`.

## Results File
**File**: `statistical_similarity_results.csv`

## Quick Interpretation Guide

### Error Thresholds

| Measure | Excellent | Good | Moderate | Poor |
|---------|-----------|------|----------|------|
| **Mean Error** | < 5% | 5-15% | 15-30% | > 30% |
| **Median Error** | < 5% | 5-15% | 15-30% | > 30% |
| **Std Error** | < 10% | 10-25% | 25-50% | > 50% |

### Models Summary

| Model | Best For | Speed | Complexity |
|-------|----------|-------|------------|
| **Gaussian Copula** | Simple datasets, quick baseline | Fast | Low |
| **CTGAN** | Complex patterns, high-dimensional data | Slow | High |
| **CopulaGAN** | Mixed data types, balanced performance | Medium | Medium |
| **TVAE** | Stable results, smooth distributions | Medium | Medium |

## CSV Columns Explained

| Column | Description |
|--------|-------------|
| `Dataset` | Bank, Cancer, or Alzhimers |
| `Model` | CTGAN, CopulaGAN, GaussianCopula, or TVAE |
| `Column` | Feature name |
| `Mean_Real` | Mean from real data |
| `Mean_Synthetic` | Mean from synthetic data |
| `Mean_Error_%` | Percentage error in mean |
| `Median_Real` | Median from real data |
| `Median_Synthetic` | Median from synthetic data |
| `Median_Error_%` | Percentage error in median |
| `Std_Real` | Standard deviation from real data |
| `Std_Synthetic` | Standard deviation from synthetic data |
| `Std_Error_%` | Percentage error in standard deviation |
| `Outliers_Real` | Number of outliers in real data |
| `Outliers_Synthetic` | Number of outliers in synthetic data |
| `Outliers_Diff` | Difference in outlier counts |

## Common Analysis Queries

### Find High Error Features
```python
import pandas as pd
results = pd.read_csv('statistical_similarity_results.csv')
high_error = results[results['Mean_Error_%'] > 30]
```

### Compare Models by Dataset
```python
results.groupby(['Dataset', 'Model'])['Mean_Error_%'].mean()
```

### Find Best Model per Dataset
```python
best = results.groupby(['Dataset', 'Model'])['Mean_Error_%'].mean().reset_index()
best.loc[best.groupby('Dataset')['Mean_Error_%'].idxmin()]
```

### Average Performance by Model
```python
results.groupby('Model')[['Mean_Error_%', 'Median_Error_%', 'Std_Error_%']].mean()
```

## Key Metrics to Watch

1. **Mean Error < 15%**: Good central tendency preservation
2. **Std Error < 25%**: Good variability preservation
3. **Outlier_Diff < 50**: Reasonable outlier preservation
4. **Consistent errors across models**: May indicate data quality issues

## Red Flags

⚠️ **High Mean Error (>30%)**: Model not capturing central tendency  
⚠️ **High Std Error (>50%)**: Model not capturing variability  
⚠️ **Large Outlier_Diff**: Model smoothing or generating unrealistic extremes  
⚠️ **Consistent failures across all models**: Possible data quality issue

## Next Steps

1. ✅ Review features with high error rates
2. ✅ Compare model performance for your specific use case
3. ✅ Validate on downstream tasks (beyond statistics)
4. ✅ Consider privacy and utility metrics
5. ✅ Test with domain experts

---
**For detailed information, see**: `STATISTICAL_SIMILARITY_RESULTS_DOCUMENTATION.md`

