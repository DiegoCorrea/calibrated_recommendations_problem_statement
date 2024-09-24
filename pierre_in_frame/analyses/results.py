from pandas import DataFrame

from settings.labels import Label


def results_by_weight(data: DataFrame, order_by_metric: str) -> dict:
    weight_dict = {}

    for index, line in data.iterrows():
        _, _, _, _, _, _, weight = line['COMBINATION'].split("-")
        weight_dict[weight].append(line[order_by_metric])

        return weight_dict


def results_by_fairness(data: DataFrame, order_by_metric: str) -> dict:
    fairness_dict = {}

    for index, line in data.iterrows():
        _, _, _, fairness, _, _, weight = line['COMBINATION'].split("-")
        if weight != "C@0.0":
            try:
                fairness_dict[fairness].append(line[order_by_metric])
            except KeyError:
                fairness_dict[fairness] = []
                fairness_dict[fairness].append(line[order_by_metric])

    return fairness_dict


def results_by_component(data: DataFrame, order_by_metric: str) -> dict:
    results_dict = {
        "Calib": [], "N-Calib": [],
        "SIM": [], "DIST": [],
        "WPS": [], "CWS": [],
        "NDCG": [], "SUM": [],
        "LOG": [], "LIN": [],
        "PERSON": [],  "CONST": []
    }

    for index, line in data.iterrows():
        _, tradeoff, distribution, fairness, relevance, selector, weight = line['COMBINATION'].split("-")
        if weight == 'C@0.0':
            results_dict["N-Calib"].append(line[order_by_metric])
            continue
        else:
            results_dict["Calib"].append(line[order_by_metric])

        if fairness in Label.SIMILARITY_LIST:
            results_dict["SIM"].append(line[order_by_metric])
        else:
            results_dict["DIST"].append(line[order_by_metric])

        if weight in Label.PERSON_WEIGHT:
            results_dict["PERSON"].append(line[order_by_metric])
        else:
            results_dict["CONST"].append(line[order_by_metric])

        results_dict[distribution].append(line[order_by_metric])
        results_dict[relevance].append(line[order_by_metric])
        results_dict[tradeoff].append(line[order_by_metric])

    return results_dict
