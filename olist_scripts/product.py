"""
This script contains methods related to an analysis to be conducted on a per
product basis for my Olist project.
"""

import pandas as pd
import numpy as np
from olist_scripts.data import Olist
from olist_scripts.order import Order

class Product:
    """
    Dataframes that have product_id as their index and a variety of features
    related to products, some engineered and some already existing.
    """
    def __init__(self):
        self.data = Olist().retrieve_data()
        self.order = Order()

    def get_listing_features(self):
        """
        Returns a df with the basic listing features of a product, like category,
        description length and name length, quantity of photos included in the
        description and the categories' translations. Also includes the product's
        size (volume in litres and mass in kg)
        """

        products = self.data["products_df"]
        transl = self.data["product_category_name_translation_df"]

        temp = products.merge(transl, how = "left", on = "product_category_name") \
            .drop(columns = ["Unnamed: 0_x", "product_category_name", "Unnamed: 0_y"])

        temp["product_mass_kg"] = temp["product_weight_g"]/1000
        temp.drop(columns = ["product_weight_g"], inplace = True)

        temp["volume_litres"] = temp["product_length_cm"]*temp["product_height_cm"] \
            *temp["product_width_cm"]/1_000
        temp.drop(columns = ["product_length_cm", "product_height_cm", "product_width_cm"], inplace = True)

        temp.rename(columns = {"product_category_name_english": "category"}, inplace = True)

        return temp

    def get_sales_features(self):
        """
        Returns a df with features related to its sales, like average price, how many
        of it were sold, number of orders that included it and total revenue because of
        it.
        """
        products = self.data["products_df"]
        items = self.data["order_items_df"]

        temp = products.merge(items, how = "left", on = "product_id")

        temp = temp.groupby(by = "product_id", as_index = False).agg({"order_id": "nunique", \
            "order_item_id": "count", "price" : "mean"})

        temp.columns = ["product_id", "n_orders", "n_items_sold", "mean_price"]

        temp["total_revenue"] = temp["n_orders"]*temp["mean_price"]

        return temp

    def get_product_review_features(self):
        """
        Returns a df with features related to the reviews for the product.
        Includes mean review score, share of one star and five star reviews.
        """

        order_reviews = self.order.get_reviews()
        products = self.data["products_df"]
        items = self.data["order_items_df"]

        temp = order_reviews.merge(items, how = "left", on = "order_id") \
            .merge(products, how = "left", on = "product_id").drop_duplicates(subset = ["order_id", "product_id"])

        temp = temp.groupby(by = "product_id", as_index = False).agg({"dim_is_five_star": "sum", "dim_is_one_star": "sum",
                                             "review_score": "mean", "order_id": "nunique"})

        temp["share_of_one_stars"] = temp["dim_is_one_star"]/temp["order_id"]
        temp["share_of_five_stars"] = temp["dim_is_five_star"]/temp["order_id"]

        return temp[["product_id", "share_of_five_stars", "share_of_one_stars", "review_score"]]

    def get_wait_time(self):
        """
        Returns a df with the average wait time per product
        """
        order_times = self.order.get_timedeltas()
        products = self.data["products_df"]
        items = self.data["order_items_df"]

        temp = order_times.merge(items, how = "left", on = "order_id") \
            .merge(products, how = "left", on = "product_id").drop_duplicates(subset = ["order_id", "product_id"])

        temp = temp.groupby(by = "product_id", as_index = False).agg({"wait_time": "mean"})

        return temp

    def get_training_data(self):
        """
        Returns a dataframe with all the per product features above combined.
        """

        return self.get_listing_features().merge(self.get_product_review_features(), on = "product_id", how = "left") \
            .merge(self.get_sales_features(), on = "product_id", how = "left") \
            .merge(self.get_wait_time(), on  = "product_id", how = "left")

    def get_data_per_category(self):
        """
        Returns a dataframe with product features related to a per category basis
        """

        temp = self.get_training_data()

        temp_cat = temp.groupby(by = "category", as_index = False).agg({"product_mass_kg": "mean", "volume_litres": "mean", \
                                           "review_score": "mean", "wait_time": "mean", \
                                           "n_orders": "sum", "n_items_sold": "sum", \
                                           "share_of_one_stars": "mean", "share_of_five_stars": "mean", \
                                           "mean_price": "mean"})

        temp_cat["total_revenue"] = temp_cat["mean_price"]*temp_cat["n_items_sold"]

        temp_cat.columns = ["category", "mean_product_mass", "mean_product_volume_litres",
                            "mean_review_score", "mean_wait_time", "total_orders",
                            "total_items_sold", "mean_share_of_one_stars", "mean_share_of_five_stars",
                            "mean_price", "total_revenue"]
        return temp_cat
