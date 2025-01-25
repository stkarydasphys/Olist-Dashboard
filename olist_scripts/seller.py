"""
Following along the previous ones, this is a script that is meant to
engineer/retrieve data that are based on the seller data for my Olist
project
"""

import numpy as np
import pandas as pd
from olist_scripts.data import Olist
from olist_scripts.order import Order
import datetime

class Seller:
    """
    Dataframes that have seller_id as their index. They have various useful
    features related to seller data.
    """

    def __init__(self):
        self.data = Olist().retrieve_data()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a df with seller_id, seller_city, seller_state
        """

        sellers = self.data["sellers_df"]

        return sellers[["seller_id", "seller_city", "seller_state"]]

    def get_seller_timedeltas(self, is_delivered = True):
        """
        Returns a df with basic timedeltas. Specifically, the average wait_time
        for this seller, the average delay_vs_expected time (negative values mean early delivery),
        the time from the seller to the carrier and from the carrier to the customer.
        By default, it calculates these only for delivered orders.
        """

        orders = self.data["orders_df"]

        if is_delivered:
            orders = orders[orders["order_status"] == "delivered"].copy()


        items = self.data["order_items_df"]
        sellers = self.data["sellers_df"]

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

        orders.loc[:,"seller_to_carrier"] = (orders["order_delivered_carrier_date"] - \
                                            orders["order_purchase_timestamp"])/one_day_delta

        orders.loc[:,"carrier_to_customer"] = (orders["order_delivered_customer_date"] - \
                                            orders["order_delivered_carrier_date"])/one_day_delta

        temp = orders.merge(items, on = "order_id", how = "left").merge(sellers, \
            on = "seller_id", how = "left")

        df = temp.groupby(by = "seller_id", as_index = False)[["wait_time", "expected_wait_time", \
            "delay_vs_expected", "seller_to_carrier", "carrier_to_customer"]].mean()

        return df

    def get_active_dates(self):
        """
        Returns a df that has as features the first sale's and last sale's data per seller
        and the total months they have been active so far on Olist.

        There is an assumption made, that for a seller to be on the platform,
        they have to be there for at least one month
        (so the minimum months on the platform is 1, not 0,
        starting from the first sale they did)
        This will of course be causing some data leakage, but it shouldn't be too significant.
        """

        orders = self.data["orders_df"]
        items = self.data["order_items_df"]
        sellers = self.data["sellers_df"]

        tmp = orders.merge(items, on = "order_id", how = "left").merge(sellers, on = "seller_id", how = "left")

        # engineering first and last orders, as well as total time on olist
        first_order = tmp.sort_values("order_purchase_timestamp").drop_duplicates(subset = "seller_id") \
            .loc[:, ["seller_id", "order_purchase_timestamp"]] \
                .rename(columns = {"order_purchase_timestamp": "first_order"}).dropna()

        last_order = tmp.sort_values("order_purchase_timestamp", ascending = False) \
            .drop_duplicates(subset = "seller_id").loc[:, ["seller_id", "order_purchase_timestamp"]] \
                .rename(columns = {"order_purchase_timestamp": "last_order"}).dropna()

        tmp = first_order.merge(last_order, how = "left", on = "seller_id")

        # creating one day datetime object to convert timedeltas into decimal numbers by dividing by it
        one_day_delta = datetime.timedelta(days=1)

        # converting dates to datetime objects
        temp_list = ["first_order", "last_order"]

        for obj in temp_list:
            tmp[obj] = pd.to_datetime(tmp[obj])

        tmp["months_on_olist"] = round((tmp.loc[:,"last_order"] - tmp.loc[:,"first_order"])/(one_day_delta*30)+1)

        return tmp

    def get_quantitative_features(self):
        """
        Returns a df that contains the total amount of orders that the seller participated in, the
        total amount of items sold by a seller and the items sold per order of the seller. Also
        returns the total revenue and the revenue per order for the seller.
        """

        orders = self.data["orders_df"]
        items = self.data["order_items_df"]
        sellers = self.data["sellers_df"]

        tmp = orders.merge(items, on = "order_id", how = "left").merge(sellers, on = "seller_id", how = "left")

        tmp = tmp.groupby(by = "seller_id", as_index = False) \
            .agg({"order_id": "nunique", "order_item_id": "sum", "price": "sum"})

        tmp.columns = ["seller_id", "order_count", "total_items_sold", "revenue"]

        tmp["items_per_order"] = tmp["total_items_sold"]/tmp["order_count"]
        tmp["revenue_per_order"] = tmp["revenue"]/tmp["order_count"]

        return tmp

    def get_review_score(self):
        """
        Returns a dataframe that has the average review per seller, and the share of 1-star and 5-star
        reviews they had.
        """

        orders = self.data["orders_df"]
        items = self.data["order_items_df"]
        sellers = self.data["sellers_df"]
        reviews = self.data["order_reviews_df"]


        tmp = orders.merge(reviews, on = "order_id", how = "left").merge(items, on = "order_id", how = "left") \
           .merge(sellers, on = "seller_id", how = "left").drop_duplicates()

        tmp["one_star"] = (tmp["review_score"] == 1).astype(int)
        tmp["five_star"] = (tmp["review_score"] == 5).astype(int)

        df = tmp.groupby(by = "seller_id", as_index = False).agg({"one_star": "sum", "five_star": "sum", \
            "review_score": "mean", "order_id": "count"})

        df.rename(columns = {"order_id": "order_count"}, inplace = True)

        df["share_of_one_stars"] = df["one_star"]/df["order_count"]
        df["share_of_five_stars"] = df["five_star"]/df["order_count"]

        return df

    def get_training_data(self):
        """
        Returns a df that has all the features related to sellers, done by the
        methods above.
        """

        return self.get_active_dates().merge(self.get_quantitative_features(), on = "seller_id", how = "left") \
            .merge(self.get_review_score(), on = "seller_id", how = "left") \
            .merge(self.get_seller_features(), on  = "seller_id", how = "left") \
            .merge(self.get_seller_timedeltas(), on = "seller_id", how = "left")
