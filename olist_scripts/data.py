"""
This is a script containing a data retrieving function for my Olist project
It should be working independently of type of machine and location
"""

import os
import pandas as pd

class Olist:
    def retrieve_data(self) -> dict:
        """
        Returns a dictionary, whose keys are the names of
        dataframes containing the olist data, and its values are
        the actual dataframes, loaded from the csv files saved in the
        data folder
        """

        # finding root directory wherein the data dir should be
        rootdir = os.path.dirname(os.path.dirname(__file__))

        # defining absolute path to the dir where the data is
        csv_path = os.path.join(rootdir, "data")

        # creating aforementioned dictionary
        file_names = [name for name in os.listdir(csv_path) if name[-4:] == ".csv"]
        key_names = [name.replace(".csv", "_df") for name in file_names]

        data = {key_name: pd.read_csv(os.path.join(csv_path, file)) \
                for (key_name, file) in zip (key_names, file_names)}

        return data
