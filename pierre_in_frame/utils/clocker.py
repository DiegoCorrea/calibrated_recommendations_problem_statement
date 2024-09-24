import datetime
import logging
import time

from pandas import DataFrame

logger = logging.getLogger(__name__)


class Clocker:
    """
    It is a generic class to lead with the execution time.
    """

    def __init__(self):
        """
        It is a generic class to lead with the execution time.
        """
        self.start_time = None
        self.finish_time = None
        self.start_time_to_show = None
        self.finish_time_to_show = None
        self.time_data_df = None

    def start_count(self):
        """
        This method starts the time counter.
        """
        self.start_time = time.time()
        self.start_time_to_show = time.strftime('%H:%M:%S')

    def get_start_time(self):
        """
        This method returns the start time counter value.
        """
        return self.start_time

    def get_start_time_to_show(self):
        """
        This method returns the start time counter value.
        """
        return self.start_time_to_show

    def finish_count(self):
        """
        This method finishes the time counter value.
        """
        self.finish_time = time.time()
        self.finish_time_to_show = time.strftime('%H:%M:%S')

    def get_finish_time(self):
        """
        This method returns the time that the process is finished.
        """
        return self.finish_time

    def get_finish_time_to_show(self):
        """
        This method returns the time that the process is finished.
        """
        return self.finish_time_to_show

    def get_total_time(self):
        """
        This method returns the process total time.
        """
        return self.finish_time - self.start_time

    def get_total_time_formatted(self):
        """
        This method returns the process total time.
        """
        return str(datetime.timedelta(seconds=self.finish_time - self.start_time))

    def clock_data(self) -> DataFrame:
        """
        This method creates a dataframe with the process times and returns it.
        """
        total_time = datetime.timedelta(seconds=self.get_total_time())
        self.time_data_df = DataFrame({
            "stated_at": [self.get_start_time()],
            "finished_at": [self.get_finish_time()],
            "total": [total_time]
        })
        return self.time_data_df

    def print_time_info(self):
        string_tp_print = "...".join([
            'ooo start at ' + self.get_start_time_to_show(),
            ' XXX stop at ' + self.get_finish_time_to_show(),
            ' $$$ total time ' + self.get_total_time_formatted(),
        ])
        logger.info(string_tp_print)
