import datetime
import logging
import os
import platform
import socket
import time

from pandas import DataFrame
from settings.constants import Constants

logger = logging.getLogger(__name__)


class Step:
    """
    It is a generic class to lead with the steps.
    """

    def __init__(self):
        """
        It is a generic class to lead with the steps.
        """
        self.experimental_settings = None
        self.start_time = None
        self.finish_time = None
        self.time_data_df = None

    def read_the_entries(self):
        """
        This method is to be overridden in each step class.
        """
        pass

    def set_the_logfile(self):
        """
        This method is to be overridden in each step class.
        """
        pass

    def print_basic_info(self):
        """
        This method is to be overridden in each step class.
        """
        pass

    @staticmethod
    def machine_information():
        """
        This method prints the main computer information.
        """
        node = '' or platform.node() or socket.gethostname() or os.uname().nodename
        logger.info("> MACHINE INFORMATION")
        logger.info(" ".join(['>>', 'N Cores:', str(Constants.N_CORES)]))
        logger.info(" ".join(['>>', 'Machine RAM:', str(Constants.MEM_RAM)]))
        logger.info(" ".join(['>>', 'Machine Name:', node]))

    def start_count(self):
        """
        This method starts the time counter.
        """
        self.start_time = time.time()
        logger.info('ooo start at ' + time.strftime('%H:%M:%S'))

    def get_start_time(self):
        """
        This method returns the start time counter value.
        """
        return self.start_time

    def finish_count(self):
        """
        This method finishes the time counter value.
        """
        self.finish_time = time.time()
        logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))

    def get_finish_time(self):
        """
        This method returns the time that the process is finished.
        """
        return self.finish_time

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
            'ooo start at ', str(datetime.timedelta(seconds=self.get_start_time())),
            ' XXX stop at ', str(datetime.timedelta(seconds=self.get_finish_time())),
            ' $$$ total time ', str(datetime.timedelta(seconds=self.get_total_time())),
        ])
        logger.info(string_tp_print)
