import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# Load data.
data_x = pd.read_csv('vehmpgdata.csv')

# Drop NaNs.
data_x.dropna(how='any', inplace=True)

# I hate this hard-coding, but I'll go with what the prompt says.
x_train = data_x.iloc[:, 0:18].values
y_train = data_x.iloc[:, 18].values

# Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    test_size=0.2,
                                                    random_state=1953)

# Use DecisionTreeClassifier.
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
# Compute f1 score and formulate the confusion matrix.
# TODO: choose best 'average' parameter.
dtc_f1 = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
dtc_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Decision Tree Classifier F1 Score: {:.4f}'.format(dtc_f1))
print('Decision Tree Classifier Confusion Matrix:')
print(dtc_cm)

# Export to file.
# export_graphviz(dtc, out_file='vehicle.dot')

# Random Forest:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=3,
                                   random_state=1953))
])
# Train.
pipeline.fit(x_train, y_train)
# Predict.
y_pred = pipeline.predict(x_test)
rf_f1 = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
rf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Random Forest Classifier F1 Score: {:.4f}'.format(rf_f1))
print('Random Forest Classifier Confusion Matrix:')
print(rf_cm)

importances = pipeline.named_steps['clf'].feature_importances_
indices = np.argsort(importances)[::-1]
# print feature importances in descending order
for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot bar graph of feature importances

# plt.figure()
# plt.title("Feature importances using Random Forest Classifier")
# plt.xlabel('Number of feature')
# plt.ylabel('Importance')
# plt.bar(range(x_train.shape[1]), importances[indices],
#         color="b", align="center")
# plt.show()

# Visualize highly correlated features

# sns.pairplot(data_x, dropna=True)
# plt.show()

# PCA
pipeline = Pipeline([('scaling', StandardScaler()), ('pca',
                                                     PCA(n_components=2))])
x2d = pipeline.fit_transform(x_train)
x2dt = pipeline.transform(x_test)

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', DecisionTreeClassifier(max_depth=3))
])

# Train.
pipeline.fit(x2d, y_train)

# Test.
y_pred = pipeline.predict(x2dt)

pca_f1 = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
pca_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('PCA + Tree F1 Score: {:.4f}'.format(pca_f1))
print('PCA + Tree Confusion Matrix:')
print(pca_cm)

# export_graphviz(pipeline.named_steps['clf'],
#                 out_file='vehiclePCA.dot')

# Random Forest on PCA features.
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=3,
                                   random_state=1953))
])
# Train.
pipeline.fit(x2d, y_train)
# Test.
y_pred = pipeline.predict(x2dt)

pca_rf_f1 = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
pca_rf_cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('PCA + RF F1 Score: {:.4f}'.format(pca_rf_f1))
print('PCA + RF Confusion Matrix:')
print(pca_rf_cm)

#Set min and max values and give it some padding
x_min, x_max = x2dt[:, 0].min() - .5, x2dt[:, 0].max() + .5
y_min, y_max = x2dt[:, 1].min() - .5, x2dt[:, 1].max() + .5
h = 0.1
print(x_min, x_max, y_min, y_max)
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole grid
Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plot the contour and test examples
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(x2dt[:, 0], x2dt[:, 1], c=y_test, cmap=plt.cm.Spectral)
plt.savefig('PCARandForest.png')

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(x2d[:, 0], x2d[:, 1], c=y_train, cmap=plt.cm.Spectral)
plt.savefig('PCARandForest2.png')
pass