# Olist Dashboard

This is a solo-project, meant for practice on all things Python related to data analysis, data visualization and
elementary ML. It is based on the public datasets by Olist found on [Kaggle](https://www.kaggle.com/datasets/terencicp/e-commerce-dataset-by-olist-as-an-sqlite-database).

This project serves multiple purposes:
- Improving Python and Data Science skills
- Practice Git and project structuring
- Be the foundation for a future frontend dashboard
- Offer the basis for a future sentiment anlysis based on reviews provided for the prodcuts/sellers.

To ensure scalability I have added scripts to return the dataframes (or ones similar to them) that I have used throughout the notebook(s). The project also includes basic regressions to explore and possibly support some of the findings of the EDA
done in the notebooks.

Finally, it includes a basic what-if analysis to increase profits, based on a billing paradigm proposed by the Le Wagon
bootcamp in the corresponding project we did there.

** Disclaimer **
As already mentioned, this dataset was proposed to be worked on through our first project week in the Le Wagon bootcamp.
The work included here is entirely my own, with the exception of the billing paradigm mentioned, which follows the methodology proposed in the bootcamp. Everything is done from scratch, though I do follow the general skeleton/rationale of the project, that was originally proposed in the bootcamp.

## Installation

In the bash terminal copy and execute the following:
```sh
git clone https://github.com/stkarydasphys/Olist-Dashboard
cd Olist-Dashboard
pip install -r requirements.txt
```

## Project Structure

├── EDA_notebooks/
│   ├── Clean EDA.ipynb
│   ├── Creating pandas dfs from sqlite file.ipynb
├── olist_scripts/
|   ├── data.py
|   ├── order.py
|   ├── product.py
|   ├── review.py
|   ├── seller.py
├── requirements.txt
├── README.md
├── .gitignore

## Usage

All EDA performed with pandas as well as some regressions that were performed are situated within EDA_notebooks/Clean EDA.ipynb.
A main script to orchestrate the olist_scripts/ will eventually be added. Until then, methods can be accessed manually.
For example, in a Jupyter notebook, to retrieve all the data in a data dictionary:

```python
from olist_scripts.data import Olist

data = Olist.retrieve_data()
```

and create an Orders object which is the main dataframe:

```python
from olist_scripts.order import Order
orders = Order()

# get the full dataframe, including engineered features
orders.get_training_data()
```
