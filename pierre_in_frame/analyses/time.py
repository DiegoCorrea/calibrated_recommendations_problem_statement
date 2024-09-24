from pandas import DataFrame
import datetime


def execution_time_analyze(data: DataFrame):
    print("Total Time: ", str(datetime.timedelta(seconds=data['TIME'].sum())))
