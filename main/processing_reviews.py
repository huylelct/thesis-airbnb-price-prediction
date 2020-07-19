
  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def normalize(x):
    x_train_normalized = x.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x_train_normalized)

    # Run the normalizer on the dataframe
    return pd.DataFrame(x_scaled, columns=x.columns)


def split(data, val_frac=0.10, test_frac=0.10):
    x = data.loc[:, data.columns != 'price']
    x = x.loc[:, x.columns != 'id']
    x = x.loc[:, x.columns != 'Unnamed: 0']

    y = data['price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(val_frac + test_frac), random_state=1)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_frac / (val_frac + test_frac),
                                                    random_state=1)

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    data = pd.read_csv('../data/listings_cleaned.csv')

    x_train, y_train, x_val, y_val, x_test, y_test = split(data)

    # x_train = normalize(x_train)
    # x_val = normalize(x_val)
    # x_test = normalize(x_test)

    x_train.to_csv('../data/data_cleaned_train_X.csv', header=True, index=False)
    y_train.to_csv('../data/data_cleaned_train_y.csv', header=True, index=False)

    x_val.to_csv('../data/data_cleaned_val_X.csv', header=True, index=False)
    y_val.to_csv('../data/data_cleaned_val_y.csv', header=True, index=False)

    x_test.to_csv('../data/data_cleaned_test_X.csv', header=True, index=False)
    y_test.to_csv('../data/data_cleaned_test_y.csv', header=True, index=False)

    pass
