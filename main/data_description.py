import pandas as pd
import math

# import data set
filename = '../data/airbnb-hcm-1.csv'
data = pd.read_csv(filename)

data = data[data.columns.intersection(['longitude', 'latitude', 'address'])]
print(data)
# for feature in data.describe().columns:
#     print(feature, data.describe()[feature])
