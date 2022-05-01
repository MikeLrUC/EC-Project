import math
import numpy as np
import pandas as pd
import scipy.stats as st

from .plotter import Plotter
from .logger import Logger

class Statistics:
    """
    | Statistics ══╦══ Descriptive ══╦══ Centrality ══╦══ Mean
    |              ║                 ║                ╠══ Median
    |              ║                 ║                ╚══ Mode
    |              ║                 ║
    |              ║                 ╠══ Spread ══╦══ Standard Deviation        
    |              ║                 ║            ╚══ Variance
    |              ║                 ║
    |              ║                 ╠══ Shape ══╦══ Skewness
    |              ║                 ║           ╚══ Kurtosis
    |              ║                 ║
    |              ║                 ╚══ Center & Spread ══╦══ Minimum      (0%)
    |              ║                                       ╠══ Q1           (25%)
    |              ║                                       ╠══ Q2 / Median  (50%)
    |              ║                                       ╠══ Q3           (75%)
    |              ║                                       ╚══ Maximum      (100%)
    |              ║
    |              ╚══ Inference ══╦══ Normality ══╦══ Kolmogorov-Smirnov test
    |                              ║               ╠══ Shapiro-Wilk test
    |                              ║               ╚══ Anderson-Darling test
    |                              ║               
    |                              ╠══ Variance Homogeneity ═════ Levene Test
    |                              ║               
    |                              ╠══ Effect Size ══╦══ Parametric ═════ T test Effect Size 
    |                              ║                 ║
    |                              ║                 ╚══ Non Parametric ══╦══ Mann Whitney Effect Size 
    |                              ║                                      ╚══ Wilcoxon Effect Size
    |                              ║               
    |                              ╚══ Hypothesis Testing ══╦══ Parametric ══╦══ Paired ══╦══ Matched ═════ Dependent T test
    |                                                       ║                ║            ╚══ Unmatched ═════ Independent T test
    |                                                       ║                ║ 
    |                                                       ║                ╚══ Multiple ══╦══ Matched ═════ One-Way repeated measures ANOVA
    |                                                       ║                               ╚══ Unmatched ═════ One-Way ANOVA
    |                                                       ║
    |                                                       ╚══ Non Parametric ══╦══ Paired ══╦══ Matched ═════ Wilcoxon test
    |                                                                            ║            ╚══ Unmatched ═════ Mann-Whitney test
    |                                                                            ║ 
    |                                                                            ╚══ Multiple ══╦══ Matched ═════ Friedman test
    |                                                                                           ╚══ Unmatched ═════ Kruskal-Wallis test
    """

    #-*-# Util #-*-#

    def load(df: pd.DataFrame):
        return df.to_numpy().transpose()

    def visual_analysis(data, normal=True, show=True):
        figures = []
        for i in range(len(data)):
            figures.append(Plotter.histogram(Statistics.normalize(data[i]), f"Group {i}", normal=normal, show=show))
        return figures

    def normalize(data): # https://www.statisticshowto.com/sigma-sqrt-n-used/
        return (data - np.mean(data)) / (np.std(data) / np.sqrt(len(data)))

    def effect_size_non_parametric(z_score, n_observations):
        return z_score / math.sqrt(n_observations)
    
    #-*-# Descriptive Statistics #-*-#

    def describe(data):
        """
        - The minimum is the smallest value in the data set.
        - The maximum is the largest value in the data set.
        - The mean summarizes an entire dataset with a single number representing the data's center point or typical value.
        - The standard deviation is a measure of the amount of variation or dispersion of a set of values
        - The variance measures variability from the average or mean.
        - Variance is the average squared deviations from the mean, while standard deviation is the square root of this number. 
            Both measures reflect variability in a distribution, but their units differ: 
            Standard deviation is expressed in the same units as the original values
        - The median is the value separating the higher half from the lower half of a data sample.
        - The mode is the value that appears most often in a set of data values.
        - The Skewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point. 
        - The Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution.
        - A quartile is a type of quantile which divides the number of data points into four parts, or quarters, of more-or-less equal size.
        """
        description = {
            "Minimum": np.min(data),
            "Maximum": np.max(data),
            "Mean": np.mean(data),
            "Standard Deviation": np.std(data),
            "Variance": np.var(data),
            "Median": np.median(data),
            "Mode": st.mode(data),
            "Skew": st.skew(data),
            "Kurtosis": st.kurtosis(data),
            "Q1 - Q2 - Q3": np.percentile(data, [25, 50, 75]),
        }
        return description 

    #-*-# Inference Statistics #-*-#

    # Normality Tests

    def kolmogorov_smirnov(data):
        return st.kstest(Statistics.normalize(data), 'norm')

    def shapiro_wilk(data):
        return st.shapiro(Statistics.normalize(data))

    def anderson_darling(data):
        return st.anderson(Statistics.normalize(data))


    # Variance Homogeneity Tests

    def levene(data):
        return st.levene(*data)
    

    #-*-# Hypothesis Testing #-*-#

    ## Parametric
    def t_test(data1, data2, independent: bool):
        if independent:
            return st.ttest_ind(data1, data2)
        return st.ttest_rel(data1, data2)

    def one_way_anova(data, independent: bool):
        if independent:
            return st.f_oneway(*data)
        #TODO: Repeated Measures ANOVA

    ## Non-Parametric

    def mann_whitney(data1, data2):
        return st.mannwhitneyu(data1, data2)

    def wilcoxon(data1, data2):
        return st.wilcoxon(data1, data2)

    def kruskal_wallis(data):
        return st.kruskal(*data)

    def friedman(data):
        return st.friedmanchisquare(*data)
    
    #-*-# Effect Size [Only for paired Comparisons] #-*-# #TODO: Integrate these
    
    # Pearson Correlation Coefficient
    def effect_size_t_test(test_statistic, n, independent: bool):
        df = 2 * (n - 1) if independent else n - 1
        return math.sqrt(test_statistic ** 2) / math.sqrt(df + test_statistic ** 2)

    def effect_size_mann_whitney(test_statistic, n1, n2, n_observations):
        z_score = (test_statistic - (n1 * n2) / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        return Statistics.effect_size_non_parametric(z_score, n_observations)
    
    def effect_size_wilcoxon(test_statistic, n, n_observations):
        z_score = (test_statistic - (n * (n + 1) / 4)) / math.sqrt(n * (n + 1) * (2*n + 1) / 24)
        return Statistics.effect_size_non_parametric(z_score, n_observations)

    #-*-# Analysis #-*-#

    def analyse(df: pd.DataFrame, matched: bool, filename=None, alpha=0.05):
        samples = len(df.columns)

        # Load Data
        data = Statistics.load(df)

        # Visual Analysis
        figures = Statistics.visual_analysis(data, show=False)
        if filename:
            Logger.save_figures(figures, filename)

        if samples > 1:
            paired = True if samples == 2 else False

            # Parametric Assumptions Testing

            # Normality
            p_values = []
            for i in range(samples):
                ks, ks_p = Statistics.kolmogorov_smirnov(data[i])
                sw, sw_p = Statistics.shapiro_wilk(data[i])
                #ad, ad_p = Statistics.anderson_darling(data[i])
                #p_values += [ks_p, sw_p, ad_p]
                p_values += [ks_p, sw_p]

            # Variance Homogeneity
            lev, lev_p = Statistics.levene(data)
            p_values.append(lev_p)

            #FIXME: Check if this is right
            parametric = False if min(p_values) < alpha else True

            # Hypothesis Testing

            if parametric:
                if paired:
                    if matched:
                        result = Statistics.t_test(data[0], data[1], independent=False)
                    else:
                        result = Statistics.t_test(data[0], data[1], independent=True)
                else:
                    if matched:
                        result = Statistics.one_way_anova(data, independent=False)
                    else:
                        result = Statistics.one_way_anova(data, independent=True)
            else:
                if paired:
                    if matched:
                        result = Statistics.wilcoxon(data[0], data[1])
                    else:
                        result = Statistics.mann_whitney(data[0], data[1])
                else:
                    if matched:
                        result = Statistics.friedman(data)
                    else:
                        result = Statistics.kruskal_wallis(data)
            return result
        return None