import pandas as pd
import numpy as np
from olist_scripts.data import Olist
import datetime

class Order:
    """
    Dataframes that have order_id as index and various properties of the orders as columns.
    So far contains methods that create dataframes with useful timedeltas and review related columns.
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

    def get_reviews(self):
        """
        Returns dataframe that contains a per order_id review related row. The dataframe has a 0 or 1 mask
        for orders that have 5 or 1 as their score (dim_is_five_star, dim_is_one_star).
        Also returns the actual review score (review_score) and the review comment (review_all).
        If no review was left, the corresponding column contains "no review"
        """
        reviews = self.data["order_reviews"].copy()

        # creating 0 or 1 columns if review score is 1 or 5
        reviews.loc[:,"dim_is_five_star"] = reviews["review_score"].apply(lambda x: 1 if x == 5 else 0)
        reviews.loc[:,"dim_is_one_star"] = reviews["review_score"].apply(lambda x: 1 if x == 1 else 0)

        # merging review title and comment
        reviews.loc[:,"review_all"] = reviews["review_comment_title"].add(reviews["review_comment_message"], fill_value = "")

        # filling nulls
        reviews.loc[reviews["review_all"].isna(), "review_all"] = "no review"

        return reviews[["order_id", "dim_is_five_star", "dim_is_one_star", "review_score", "review_all"]]

    def get_num_of_items(self):
        """
        Returns a dataframe that contains a per order id total number of items included.
        """
        items = self.data["order_items"].copy()

        # summing to find the total items per order
        return items.groupby(by = "order_id").agg({"order_item_id":"sum"}) \
            .reset_index().rename(columns = {"order_item_id": "number_of_items"})


    def get_num_sellers(self):
        """
        Returns a dataframe that contains a per order id total number of sellers included
        """
        items = self.data["order_items"].copy()

        # counting unique sellers per order id
        num_of_sellers = items.groupby(by = ["order_id"]).agg({"seller_id": pd.Series.nunique})

        return num_of_sellers.reset_index().rename( \
            columns = {"seller_id":"number_of_sellers"})

    def get_revenue_and_freight(self):
        """
        Returns the total revenue and freight value related to each order_id
        """
        items = self.data["order_items"].copy()

        # summing total revenue for the seller and freight cost
        rev_freight = items.groupby("order_id").agg({"price":"sum", "freight_value": "sum"})
        return rev_freight.reset_index().rename(columns = {"price": "revenue"})

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a dataframe with no null values, that contain all the columns above, namely:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score', 'review_all',
        'number_of_items', 'number_of_sellers', 'revenue', 'freight_value',
        'distance_seller_customer'] (distance from seller to customer is still WiP)
        """

        return self.get_wait_time(is_delivered).merge(self.get_review_score(), on = "order_id") \
            .merge(self.get_number_items(), on = "order_id") \
            .merge(self.get_number_sellers(), on = "order_id") \
            .merge(self.get_revenue_and_freight(), on = "order_id").dropna()
