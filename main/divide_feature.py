import pandas as pd

# data = pd.read_csv("../data/data_cleaned.csv")
filename = '../data/data_cleaned.csv'
data = pd.read_csv(filename)

all_columns = data.columns
feature_numeric = [
    'score',
    'reviews',
    'guest',
    'bedroom',
    'bed',
    'bath',
    'score_cleanliness',
    'score_accuracy',
    'score_communication',
    'score_location',
    'score_check_in',
    'score_value',
    'service_fee',
    'host_reviews',
    'host_response_rate',
    'cleaning_fee',
    'shared_bath',
    'private_bath',
    'price'
]

data_numeric = data[data.columns.intersection(feature_numeric)]

feature_numeric.remove('price')
data_category = data.drop(columns=feature_numeric)

data_category.to_csv("../data/data-category.csv")
data_numeric.to_csv("../data/data_numeric.csv")
