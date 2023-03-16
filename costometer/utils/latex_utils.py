import numpy as np


def get_pval_text(p_val):
    if p_val < 0.001:
        ptext = "p < 0.001"
    elif p_val < 0.01:
        ptext = "p < 0.01"
    elif p_val > 0.05:
        ptext = "p > 0.05"
    else:
        ptext = f"p = {p_val:.2f}"
    return ptext


def get_mann_whitney_text(comparison):
    return (
        f"Mann-Whitney $U={comparison['U-val'].values[0]:.2f}, "
        f"{get_pval_text(comparison['p-val'].values[0])}$, "
        f"{comparison['alternative'].values[0]}"
    )


def get_wilcoxon_text(comparison):
    return (
        f"$W = {comparison['W-val'].values[0]:.2f}, "
        f"RBC = {comparison['RBC'].values[0]:.2f},"
        f" {get_pval_text(comparison['p-val'].values[0])}$, "
        f"{comparison['alternative'].values[0]}"
    )


def get_kruskal_wallis_text(comparison):
    return (
        f"Kruskal-Wallis $H({comparison['ddof1'].values[0]})"
        f" = {comparison['H'].values[0]:.2f}, "
        f"{get_pval_text(comparison['p-unc'].values[0])}$"
    )


def get_correlation_text(correlation):
    if np.any(np.isnan(correlation["CI95%"][0])):
        return "Potentially no variance of one parameter..."

    if "pearson" in correlation.index.values:
        return (
            f"$r({correlation['n'].values[0]}) = {correlation['r'].values[0]:.2f}, "
            f"{get_pval_text(correlation['p-val'].values[0])}, "
            f"95\%\ C.\ I.\: "  # noqa: W605
            f"[{correlation['CI95%'][0][0]:.2f}, {correlation['CI95%'][0][1]:.2f}]$"
        )
    elif "spearman" in correlation.index.values:
        return (
            f"Spearman's $\\rho({correlation['n'].values[0]}) ="
            f" {correlation['r'].values[0]:.2f}, "
            f"{get_pval_text(correlation['p-val'].values[0])}, "
            f"95\%\ C.\ I.\: "  # noqa: W605
            f"[{correlation['CI95%'][0][0]:.2f}, {correlation['CI95%'][0][1]:.2f}]$"
        )
    else:
        raise NotImplementedError


def get_ttest_text(comparison):
    if isinstance(comparison["dof"].values[0], int) or isinstance(
        comparison["dof"].values[0], np.int64
    ):
        dof = comparison["dof"].values[0]
    else:
        dof = f"{comparison['dof'].values[0]:.2f}"
    return (
        f"$t({dof}) = {comparison['T'].values[0]:.2f},"
        f" {get_pval_text(comparison['p-val'].values[0])}$"
    )


def get_regression_text(regression_res):
    return (
        f"adj. $R^2={regression_res.rsquared_adj:.2f}, "
        f"F({regression_res.df_model:.0f}, {regression_res.df_resid:.0f}) = "
        f"{regression_res.fvalue:.2f}, {get_pval_text(regression_res.f_pvalue)}$"
    )


def get_anova_text(anova_object):
    anova_text = ""
    for row_idx, row in anova_object.iterrows():
        anova_text = (
            anova_text + f"{row['Source']}: $F({row['DF1']}, {row['DF2']}) ="
            f" {row['F']:.2f}, {get_pval_text(row['p-unc'])}$\n"
        )
    return anova_text
