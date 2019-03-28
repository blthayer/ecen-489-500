import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Avoid truncation when using 'describe'
pd.set_option('display.max_columns', None)

########################################################################
# Step 1): Load data.
########################################################################
df = pd.read_csv('ecen489py3data.csv')

########################################################################
# Step 2): Text attributes to numeric attributes.
########################################################################

# Code from Prof. Nowka to cast 'object' data types to categorical.
for col_name in df.columns:
    if df[col_name].dtype == 'object':
        df[col_name] = df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

# Step 3): Fix bad observations.
# Find and drop bad data value.
bad_idx = df["Feature07"].idxmax()
df.drop(bad_idx, inplace=True)

########################################################################
# Step 4): Fix NaNs.
########################################################################

# Feature01 is a data collection issue, drop NaNs.
df.dropna(subset=['Feature01'], inplace=True)

# Features 0, 2, and 3 should have NaN's zeroed out. I think this step
# is unnecessary.
zero_cols = {'Feature00': 0, 'Feature02': 0, 'Feature03': 0}
df.fillna(zero_cols, inplace=True)

# Fill other NaN's. The assignment says to do this only after having
# fixed features 00-03, but I'm not convinced it matters (besides
# dropping rows with NaNs in Feature01).
df.fillna(0, inplace=True)

########################################################################
# Step 5 (labeled 3 in the prompt) - divide into training, testing,
# validation.
########################################################################

# Separate labels and data.
x = df.drop('Label', axis=1).values
y = df['Label'].values

# Split into train and test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=0)

# Further split to get validation data.
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5,
                                                random_state=0)

########################################################################
# Step 6 (labeled 4 in the prompt) - Normalize.
########################################################################
# Initialize scaler object.
scaler = StandardScaler()
# Fit to the training data, then transform it.
x_train = scaler.fit_transform(x_train)
# Transform testing and validation data.
# TODO: Prompt mentions there's some work to do before final test
#   evaluation.
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

########################################################################
# Step 7 (labeled 5 in prompt) - Classify w/ Naive Bayes.
########################################################################
# Initialize and train.
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

# Predict.
y_pred_nb = nb_classifier.predict(x_val)

# Generate confusion matrix.
cm_nb = confusion_matrix(y_true=y_val, y_pred=y_pred_nb)

# Display.
print(cm_nb)
print('Naive Bayes training accuracy: {:.4f}'.format(nb_classifier.score(
    x_train, y_train)))
print('Naive Bayes testing accuracy: {:.4f}'.format(nb_classifier.score(
    x_val, y_val)))
pass