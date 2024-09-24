from pandas import DataFrame

from analyses.results import results_by_fairness


def best_and_worst_systems(data: DataFrame, order_by_metric: str, ascending: bool):
    print_results = data[[order_by_metric, "COMBINATION"]].sort_values(by=[order_by_metric], ascending=ascending)
    print_results.reset_index(inplace=True)
    print("-" * 30)
    print("-" * 30)
    print("-" * 30)
    print("Top-10 Best Systems on " + order_by_metric)
    print(print_results.head(10))
    print("*" * 30)
    print("Top-10 Worst Systems on " + order_by_metric)
    print(print_results.tail(10))
    print("-" * 30)
    print("-" * 30)
    print("-" * 30)


def best_and_worst_fairness_measure(data: DataFrame, order_by_metric: str, ascending: bool):
    from_f = results_by_fairness(data=data, order_by_metric=order_by_metric)
    fairness_data = DataFrame.from_dict(from_f)
    a = DataFrame([[measure, value] for measure, value in zip(fairness_data.columns.tolist(), fairness_data.mean().tolist())], columns=["measure", "value"])
    print_results = a.sort_values(by=["value"], ascending=ascending)
    print_results.reset_index(inplace=True)
    print("-" * 30)
    print("-" * 30)
    print("-" * 30)
    print("Top-10 Best Fairness Measures on " + order_by_metric)
    print(print_results.head(5))
    print("*" * 30)
    print("Top-10 Worst Fairness Measures on " + order_by_metric)
    print(print_results.tail(5))
    print("-" * 30)
    print("-" * 30)
    print("-" * 30)
