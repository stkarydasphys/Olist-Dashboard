import pandas as pd
import numpy as np
from olist_scripts.data import Olist
import datetime

class Order:
    """
    Dataframes that have order_id as index and various properties
    of the orders as columns
    """

    def __init__(self):
        self.data = Olist.retrieve_data()

    def get_timedeltas(self, is_delivered = True):
        """
        Filters only delivered orders, unless otherwise stated by parameter
        is_delivered.
        Converts string-type, date-containing columns into datetime objects.
        Then, calculates various timedeltas as decimal numbers. Specifically,
        calculates total wait time (wait_time), predicted wait time for the
        order to be delivered (expected_wait_time) and how much time the
        prediction was off (delay_vs_expected, 0 if it got there early!)
        """
        orders = self.data["orders"].copy()

        if is_delivered:
            orders = orders[orders["order_status"] == "delivered"]

        # converting dates to datetime objects
        temp_list = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
             "order_delivered_customer_date", "order_estimated_delivery_date"]

        for obj in temp_list:
            orders[obj] = pd.to_datetime(orders[obj])

        # creating one day datetime object to convert timedeltas into decimal numbers by dividing by it
        one_day_delta = datetime.timedelta(days=1)

        # calculating new series
        orders.loc[:,"wait_time"] = (orders["order_delivered_customer_date"] - \
            orders["order_purchase_timestamp"])/one_day_delta

        orders.loc[:,"expected_wait_time"] = (orders["order_estimated_delivery_date"] - \
                                            orders["order_purchase_timestamp"])/one_day_delta

        orders.loc[:,"delay_vs_expected"] = (orders["order_delivered_customer_date"] - \
                                            orders["order_estimated_delivery_date"])/one_day_delta

        # turning negative delays into 0
        orders["delay_vs_expected"] = orders["delay_vs_expected"].apply(lambda x: 0 if x < 0 else x)

        return orders[["order_id", "wait_time", "expected_wait_time", "delay_vs_expected", "order_status"]]
