from pandas import DataFrame
from scipy.stats import ttest_ind

from analyses.results import results_by_component
from settings.labels import Label


def welch(data, order_by_metric, ascending):
    p_value = 0.05
    results = results_by_component(data, order_by_metric)

    print("-" * 30)
    print("-" * 30)
    print("-" * 30)
    print("Welch Test using " + order_by_metric)

    calib_system = ttest_ind(a=results['N-Calib'], b=results['Calib'], equal_var=False, alternative='two-sided')
    print("N-Calib X Calib -> statistic: " + str(calib_system[0]) + " || -> pvalue: " + str(round(calib_system.pvalue, 5)))
    # if calib_system.pvalue < p_value:
    #     print("N-Calib < Calib")
    # else:
    #     print("N-Calib == Calib")

    cws_wps = ttest_ind(a=results['CWS'], b=results['WPS'], alternative='two-sided')
    print("CWS X WPS -> statistic: " + str(cws_wps[0]) + " || -> pvalue: " + str(round(cws_wps[1], 5)))
    # if cws_wps[1] < p_value:
    #     print("CWS < WPS")
    # else:
    #     print("CWS == WPS")

    div_sim = ttest_ind(a=results['DIST'], b=results['SIM'], alternative='two-sided')
    print("DIST X SIM -> statistic: " + str(div_sim[0]) + " || -> pvalue: " + str(round(div_sim[1], 5)))
    # if div_sim[1] < p_value:
    #     print("DIST < SIM")
    # else:
    #     print("DIST == SIM")

    sum_ndcg = ttest_ind(a=results['SUM'], b=results['NDCG'], alternative='two-sided')
    print("SUM X NDCG -> statistic: " + str(sum_ndcg[0]) + " || -> pvalue: " + str(round(sum_ndcg[1], 5)))
    # if sum_ndcg[1] < p_value:
    #     print("SUM < NDCG")
    # else:
    #     print("SUM == NDCG")

    lin_log = ttest_ind(a=results['LIN'], b=results['LOG'], alternative='two-sided')
    print("LIN X LOG -> statistic: " + str(lin_log[0]) + " || -> pvalue: " + str(round(lin_log[1], 5)))
    # if lin_log[1] < p_value:
    #     print("LIN < LOG")
    # else:
    #     print("LIN == LOG")

    const_person = ttest_ind(a=results['CONST'], b=results['PERSON'], alternative='two-sided')
    print("CONST X PERSON -> statistic: " + str(const_person[0]) + " || -> pvalue: " + str(round(const_person[1], 5)))
    # if const_person[1] < p_value:
    #     print("CONST < PERSON")
    # else:
    #     print("CONST == PERSON")


class WelchHypothesisTest:
    """
    TODO: Docstring
    """

    @staticmethod
    def compute_matrix_comparison(data: DataFrame, goal_label: str, goal_list: list) -> list:
        """
        TODO: Docstring
        """
        matrix = []
        for goal_a in goal_list:
            list_a = data[data[goal_label] == goal_a][Label.EVALUATION_VALUE].abs().tolist()
            temp_matrix = []
            for goal_b in goal_list:
                list_b = data[data[goal_label] == goal_b][Label.EVALUATION_VALUE].abs().tolist()
                const_person = ttest_ind(a=list_a, b=list_b)
                temp_matrix.append(round(const_person[1], 3))
            matrix.append(temp_matrix)
        return matrix

    @staticmethod
    def compare_two_systems(data: DataFrame, goal_label: str, system_a_name: str, system_b_name: str):
        system_a_list = data[data[goal_label] == system_a_name][Label.EVALUATION_VALUE].abs().tolist()
        system_b_list = data[data[goal_label] == system_b_name][Label.EVALUATION_VALUE].abs().tolist()
        const_person = ttest_ind(a=system_a_list, b=system_b_list, alternative='two-sided')
        print(
            f"{system_a_name} X {system_b_name} -> statistic: " + str(const_person[0]) + " || -> pvalue: " + str(round(const_person[1], 5)))
