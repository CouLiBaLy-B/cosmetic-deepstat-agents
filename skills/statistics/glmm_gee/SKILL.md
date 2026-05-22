---
name: glmm_gee
description: >
  Generalized linear mixed models (GLMM) and GEE for binary, count, and
  categorical longitudinal endpoints in cosmetic studies. Covers logistic
  GLMM, Poisson/negative-binomial GLMM, McNemar for 2-timepoint binary,
  and GEE as a robust alternative.
tags: [statistics, glmm, gee, binary, count, logistic, poisson, mcnemar]
---

# GLMM / GEE for non-continuous longitudinal endpoints

## When to use

| Data type   | 2 timepoints          | ≥ 3 timepoints         |
|-------------|------------------------|-------------------------|
| Binary      | McNemar's test         | Logistic GLMM / GEE     |
| Count       | Poisson/NegBin paired  | Poisson/NegBin GLMM/GEE |
| Categorical | Chi-square / Fisher    | Multinomial GEE          |

## Binary endpoints

### McNemar (2 timepoints)

For a single before/after comparison on a binary outcome (e.g.
"tolerability OK yes/no"):

```python
from statsmodels.stats.contingency_tables import mcnemar
table = [[a, b], [c, d]]  # concordant/discordant pairs
result = mcnemar(table, exact=True)
```

- Report: OR for discordant pairs, exact p-value, 95% CI.
- Effect size: Cohen's g.

### Logistic GLMM (≥ 3 timepoints)

```
logit P(Y=1) = β₀ + β_visit × visit + u_subject
```

- `family = binomial(link="logit")`.
- Random intercept per subject.
- Report: OR, 95% CI, Wald p-value.
- Fit via `statsmodels.BinomialBayesMixedGLM` or `pymer4.Lmer(family="binomial")`.

## Count endpoints

### Poisson / Negative-Binomial GLMM

For count data (e.g. number of comedones, number of complaints):

```
log E[Y] = β₀ + β_visit × visit + u_subject
```

- Check for overdispersion: if Pearson χ²/df > 1.5, switch from Poisson to
  negative-binomial.
- Report: rate ratio (exp(β_visit)), 95% CI, p-value.

## GEE as an alternative

When the marginal (population-average) effect is of interest rather than
the subject-specific effect:

- `statsmodels.GEE` with `family`, `cov_struct` (exchangeable, AR(1),
  independent).
- Use robust (sandwich) standard errors.
- Advantage: no distributional assumption on random effects.
- Disadvantage: less efficient; not ideal for subject-level prediction.

## Procedure (general)

1. **Choose** the model family based on the endpoint data type.
2. **Fit** the model.
3. **Check** residuals, overdispersion, convergence.
4. **Extract** the contrast of interest (e.g. D28 vs D0).
5. **Report** effect (OR / RR), 95% CI, p-value, effect size.
6. **Write** script + result JSON.

## Hard rules

1. **Never** apply a linear model to a binary outcome. Use logistic GLMM.
2. If overdispersion is detected, switch to negative-binomial.
3. For McNemar with expected cell count < 5, use the exact test.
4. Do not apply multiplicity here.

## References

- Zeger, S. L. & Liang, K.-Y. (1986). "Longitudinal data analysis for
  discrete and continuous outcomes." *Biometrics*, 42(1), 121–130.
- Bolker, B. M. et al. (2009). "Generalized linear mixed models: a
  practical guide for ecology and evolution." *TREE*, 24(3), 127–135.
