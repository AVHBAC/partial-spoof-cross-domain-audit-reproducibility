"""Statistical testing utilities for XPS-Forensic.

Implements:
- Bootstrap confidence intervals
- Binomial coverage test for conformal prediction validation
- Friedman test with Nemenyi post-hoc comparison
- Holm-Bonferroni correction for multiple hypothesis testing

References
----------
Efron & Tibshirani, "An Introduction to the Bootstrap", 1993.
Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets",
JMLR 7, 2006.
Vovk et al., "Algorithmic Learning in a Random World", Springer, 2005.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a summary statistic.

    Parameters
    ----------
    data : array-like, shape (n,)
        Observed samples.
    statistic : str
        One of ``'mean'`` or ``'median'``.
    confidence : float
        Confidence level, e.g. 0.95 for 95 % CI.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    lo, hi : float
        Lower and upper bounds of the confidence interval.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)

    stat_fn = {"mean": np.mean, "median": np.median}[statistic]

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
    return lo, hi


def bootstrap_eer_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for EER.

    Resamples (score, label) pairs jointly and recomputes EER on each
    resample. This is the only valid bootstrap for EER because the
    threshold-selection step must be inside the bootstrap loop.

    The ASVspoof community does not report EER CIs (only point estimates),
    so there is no established convention. This follows the standard
    percentile bootstrap (Efron & Tibshirani, 1993, Ch. 13) applied
    directly to EER as a functional of the score distribution.

    Parameters
    ----------
    scores : array-like, shape (n,)
        Detector scores (higher = more likely spoof).
    labels : array-like, shape (n,)
        Binary labels (0 = real, 1 = spoof).
    n_bootstrap : int
        Number of bootstrap resamples (2000 recommended for EER).
    confidence : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    lo, hi : float
        Lower and upper bounds of the CI on EER.
    """
    from xps_forensic.utils.metrics import compute_eer

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(scores)

    boot_eers = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_b, l_b = scores[idx], labels[idx]
        if len(np.unique(l_b)) < 2:
            continue  # skip degenerate resamples
        eer_b, _ = compute_eer(s_b, l_b)
        boot_eers.append(eer_b)

    if not boot_eers:
        return 0.0, 1.0

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_eers, 100 * alpha / 2))
    hi = float(np.percentile(boot_eers, 100 * (1.0 - alpha / 2)))
    return lo, hi


def binomial_coverage_test(
    n_covered: int,
    n_total: int,
    alpha: float,
) -> float:
    """One-sided binomial test for conformal prediction coverage.

    Tests H0: coverage >= (1 - alpha) against H1: coverage < (1 - alpha).

    Parameters
    ----------
    n_covered : int
        Number of items correctly covered by the prediction set.
    n_total : int
        Total number of items.
    alpha : float
        Target miscoverage level (e.g. 0.05 for 95 % coverage target).

    Returns
    -------
    float
        p-value for the one-sided test.  A small p-value (< 0.05)
        indicates that coverage is significantly below the target.
    """
    target_coverage = 1.0 - alpha
    # P(X <= n_covered) under Binomial(n_total, target_coverage)
    result = sp_stats.binomtest(
        n_covered, n_total, target_coverage, alternative="less"
    )
    return float(result.pvalue)


def friedman_nemenyi(
    results_matrix: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> dict:
    """Friedman test with Nemenyi post-hoc critical difference.

    Parameters
    ----------
    results_matrix : array-like, shape (n_datasets, n_algorithms)
        Performance values (higher is better).  Each row is a dataset,
        each column is an algorithm/method.

    Returns
    -------
    dict
        Keys:
        - ``friedman_stat`` : float - Friedman chi-squared statistic.
        - ``friedman_p`` : float - p-value of the Friedman test.
        - ``mean_ranks`` : ndarray, shape (n_algorithms,) - Average ranks
          (lower is better; rank 1 = best).
        - ``critical_difference`` : float - Nemenyi CD at alpha = 0.05.
    """
    results_matrix = np.asarray(results_matrix, dtype=float)
    n_datasets, k = results_matrix.shape

    # Rank each row: best gets rank 1. If higher_is_better=False,
    # smaller values are better (e.g., ECE, NLL, Brier).
    ranks = np.zeros_like(results_matrix)
    for i in range(n_datasets):
        if higher_is_better:
            order = np.argsort(-results_matrix[i])  # descending
        else:
            order = np.argsort(results_matrix[i])   # ascending
        for rank, idx in enumerate(order, start=1):
            ranks[i, idx] = rank

    mean_ranks = ranks.mean(axis=0)

    # Friedman test (pass raw data - scipy ranks internally). Orientation does not affect p-value.
    stat, p = sp_stats.friedmanchisquare(
        *[results_matrix[:, j] for j in range(k)]
    )

    # Nemenyi critical difference (alpha = 0.05)
    # q_alpha values from Demsar (2006), Table 5, for alpha=0.05
    # Using the Studentized range distribution approximation:
    # CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    # q_alpha for alpha=0.05 comes from the Studentized range distribution
    # divided by sqrt(2).  For practical purposes, use scipy's approximation.
    q_alpha = _nemenyi_critical_value(k, alpha=0.05)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    return {
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "mean_ranks": mean_ranks,
        "critical_difference": float(cd),
    }


def _nemenyi_critical_value(k: int, alpha: float = 0.05) -> float:
    """Approximate Nemenyi critical value q_alpha.

    Uses the Studentized range distribution q_{alpha, k, inf} / sqrt(2).

    Parameters
    ----------
    k : int
        Number of groups (algorithms).
    alpha : float
        Significance level.

    Returns
    -------
    float
        Nemenyi critical value.
    """
    # Studentized range quantile for k groups and infinite df
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, k, df=1e6)
    return q / np.sqrt(2)


def holm_bonferroni(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> list[bool]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Parameters
    ----------
    p_values : array-like, shape (m,)
        Raw p-values from *m* hypothesis tests.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    list of bool
        ``True`` if the corresponding null hypothesis is rejected
        (i.e. the result is significant after correction).
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    rejected = [False] * m

    for i, idx in enumerate(order):
        adjusted_alpha = alpha / (m - i)
        if p_values[idx] <= adjusted_alpha:
            rejected[idx] = True
        else:
            # Stop rejecting once we hit a non-significant result
            break

    return rejected
