import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import seaborn as sns

data_x = pd.read_csv("../data/data_cleaned_train_X.csv")
data_y = pd.read_csv("../data/data_cleaned_train_y.csv")

top = 34
bestfeatures = SelectKBest(score_func=f_regression, k=top)
fit = bestfeatures.fit(data_x, data_y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data_x.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
featureScores = featureScores.sort_values('Score', ascending=False)
print(featureScores)
np.save('../data/selected_feature.npy', [
    'bathrooms',
    'bedrooms',
    'accommodates',
    'beds',
    'review_scores_rating',
    'host_response_rate',
    'room_type',
    'host_response_time',
    'reviews_per_month',
    'latitude',
    'longitude',
    'number_of_reviews',
    'calculated_host_listings_count_private_rooms',
    'instant_bookable'
])

# print(featureScores.sort_values('Score'))

# corrmat = data_x.corr()
# top_corr_features = corrmat.index
# sns.heatmap(data_x[top_corr_features].corr(), annot=True, cmap="RdYlGn")
#
# columns = np.full((corrmat.shape[0],), True, dtype=bool)
# for i in range(corrmat.shape[0]):
#     for j in range(i + 1, corrmat.shape[0]):
#         if corrmat.iloc[i, j] >= 0.9:
#             if columns[j]:
#                 print(data_x.columns[i], data_x.columns[j])
#                 columns[j] = False
