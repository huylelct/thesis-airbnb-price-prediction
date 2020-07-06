import pandas as pd
import numpy as np
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
import seaborn as sns

data_numeric = pd.read_csv("../data/data_numeric.csv")
data = pd.read_csv("../data/data-category.csv")

top = 30
data_y = data["price"]
data_x = data.drop(columns=[
    "price",
    "Unnamed: 0",
    "Unnamed: 0.1",
    "Unnamed: 0.1.1",
])
# print(data_x.columns)
bestfeatures = SelectKBest(score_func=f_classif, k=top)
fit = bestfeatures.fit(data_x, data_y)

data_select = {}
for i in range(len(data_x.columns.tolist())):
    data_select[data_x.columns.tolist()[i]] = fit.scores_[i]

sorted_x = sorted(data_select.items(), key=operator.itemgetter(1), reverse=True)

feature_with_score = sorted_x[0:top]
for item in feature_with_score:
    print(item)

top_feature = list(map(lambda x: x[0], feature_with_score))
top_feature = top_feature + list(data_numeric.columns)
print(top_feature)

np.save('../data/selected_feature.npy', top_feature)

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
