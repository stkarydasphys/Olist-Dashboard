import pandas as pd
import numpy as np
import math
from olist_scripts.data import Olist
from olist_scripts.order import Order
from olist_scripts.product import Product


class Review:

    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.retrieve_data()
        self.order = Order()
        self.product = Product()

    def get_review_length(self):
        """
        Returns a DataFrame with:
       'review_id', 'length_review', 'review_score'
        """
        revs = self.data["order_reviews_df"]

        revs.loc[revs["review_comment_message"].isna(), "review_comment_message"] = ""
        revs["length_review"] = revs["review_comment_message"].map(lambda x: len(x))

        return revs[["review_id", "length_review", "review_score"]].drop_duplicates()

    def get_main_product_category(self):
        """
        Returns a DataFrame with:
       'review_id', 'order_id','product_category_name'
        """
        prods = self.product.get_training_data()
        revs = self.data["order_reviews_df"]
        temp = self.data["order_items_df"].merge(prods, on = "product_id", how = "left").merge(revs, on = "order_id", how = "left")
        temp = temp.rename(columns = {"category": "product_category_name"})


        return temp[["review_id", "order_id", "product_category_name"]].drop_duplicates()


    def get_training_data(self):
        """
        Returns a Dataframe with review_id as its index and all the features from the above method
        """

        return self.get_main_product_category().merge(self.get_review_length(), how = "left", on = "review_id")
