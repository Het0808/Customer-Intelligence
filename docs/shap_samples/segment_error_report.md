# SHAP Segment Error Analysis

**Model:** `XGBoost_stump_depth1`  |  **Threshold:** `0.7899`
**Test set:** 100 rows, 11 positives (11.0%)
**Overall F1:** 0.4000

---

## By Job Category

| Job | N | N+ | Prev% | F1 | Precision | Recall |
|-----|---|----|-------|-----|-----------|--------|
| entrepreneur | 6 | 1 | 16.7% | **0.0000** | 0.0000 | 0.0000 |
| housemaid | 2 | 0 | 0.0% | **0.0000** | 0.0000 | 0.0000 |
| management | 7 | 1 | 14.3% | **0.0000** | 0.0000 | 0.0000 |
| self-employed | 2 | 0 | 0.0% | **0.0000** | 0.0000 | 0.0000 |
| services | 10 | 0 | 0.0% | **0.0000** | 0.0000 | 0.0000 |
| student | 4 | 2 | 50.0% | **0.0000** | 0.0000 | 0.0000 |
| admin. | 23 | 5 | 21.7% | **0.3333** | 1.0000 | 0.2000 |
| blue-collar | 29 | 1 | 3.4% | **0.6667** | 0.5000 | 1.0000 |
| technician | 14 | 1 | 7.1% | **1.0000** | 1.0000 | 1.0000 |

## By Marital Status

| Marital | N | N+ | Prev% | F1 | Precision | Recall |
|---------|---|----|-------|-----|-----------|--------|
| divorced | 10 | 0 | 0.0% | **0.0000** | 0.0000 | 0.0000 |
| single | 35 | 7 | 20.0% | **0.2500** | 1.0000 | 0.1429 |
| married | 54 | 3 | 5.6% | **0.4000** | 0.5000 | 0.3333 |

---

## Key Findings

- **Worst job segment**: `entrepreneur` — F1=0.0000, N=6, N+=1 (16.7% positive rate).
  This segment underperforms overall F1 (0.4000) by 0.4000 points.
- **Best job segment**: `technician` — F1=1.0000.
- **Worst marital segment**: `divorced` — F1=0.0000, N=10, N+=0.

### Interpretation

Segments with F1 below the overall mean are candidates for:
1. **Threshold adjustment**: lower the decision threshold for these segments
   to improve recall at the cost of precision.
2. **Feature engineering**: add segment-specific features (e.g. interaction
   terms for job × duration or marital × campaign).
3. **Data augmentation**: if N+ is 0–1 in a segment, the model has no
   positive examples to learn from — these segments are effectively
   excluded from training signal.

### Caveat

All results are on the 100-row test split of the 500-row committed sample.
Cell sizes are too small for statistically significant conclusions.
Repeat this analysis after training on the full 41,163-row UCI dataset.