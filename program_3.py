import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# Avoid truncation when using 'describe'
pd.set_option('display.max_columns', None)

# Set figure size for plotting.
plt.rcParams['figure.figsize'] = [20, 8]

########################################################################
# Helper function for training and predicting.
########################################################################


def train_predict(model, x_train, y_train, x_test, y_test, c_str):
    """Helper for performing training and predictions.

    :param model: Initialized classifier. E.g. GaussianNb()
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_test: Testing data.
    :param y_test: Testing labels.
    :param c_str: String for classifier. E.g. "Naive Bayes"
    :returns: confusion matrix and F1 score.
    """
    # Train the model.
    model.fit(x_train, y_train)

    # Predict.
    y_pred = model.predict(x_test)

    # Generate confusion matrix.
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # Compute F1 score.
    f1 = f1_score(y_true=y_test, y_pred=y_pred)

    # Display.
    print('*' * 80)
    print(c_str)
    print('Confusion matrix:')
    print(cm)
    print('F1 score for testing data: {:.4f}'.format(f1))
    print('Training accuracy: {:.4f}'.format(model.score(x_train, y_train)))
    print('Testing accuracy: {:.4f}'.format(model.score(x_test, y_test)))
    print('*' * 80)

    return cm, f1


########################################################################
# Main method
########################################################################


def main():
    """Main function for running everything."""
    ####################################################################
    # Step 1): Load data.
    ####################################################################
    df = pd.read_csv('ecen489py3data.csv')

    ####################################################################
    # Step 2): Text attributes to numeric attributes.
    ####################################################################

    # Code from Prof. Nowka to cast 'object' data types to categorical.
    for col_name in df.columns:
        if df[col_name].dtype == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    # Step 3): Fix bad observations.
    # Find and drop bad data value.
    bad_idx = df["Feature07"].idxmax()
    df.drop(bad_idx, inplace=True)

    ####################################################################
    # Step 4): Fix NaNs.
    ####################################################################

    # Feature01 is a data collection issue, drop NaNs.
    df.dropna(subset=['Feature01'], inplace=True)

    # Features 0, 2, and 3 should have NaN's zeroed out. I think this
    # step is unnecessary since we fill the remaining columns with 0
    # afterwards.
    zero_cols = {'Feature00': 0, 'Feature02': 0, 'Feature03': 0}
    df.fillna(zero_cols, inplace=True)

    # Fill other NaN's. The assignment says to do this only after having
    # fixed features 00-03, but I'm not convinced it matters (besides
    # dropping rows with NaNs in Feature01).
    df.fillna(0, inplace=True)

    ####################################################################
    # Step 5 (labeled 3 in the prompt) - divide into training, testing,
    # validation.
    ####################################################################

    # Separate labels and data.
    x = df.drop('Label', axis=1).values
    y = df['Label'].values

    # Split into train and test.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                        random_state=0)

    # Further split to get validation data.
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=0.5,
                                                    random_state=0)

    ####################################################################
    # Step 6 (labeled 4 in the prompt) - Normalize.
    ####################################################################
    # Initialize scaler object.
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # Fit to the training data, then transform it.
    x_train = scaler.fit_transform(x_train)
    # Transform testing and validation data.
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)

    ####################################################################
    # Step 7 (labeled 5 in prompt) - Classify.
    ####################################################################
    # Initialize lists for storing results.
    f1 = []
    c_str = []

    # **NAIVE BAYES**
    # Initialize and train.
    nb_classifier = GaussianNB()
    # Train, predict, score.
    # noinspection PyTypeChecker
    cm_nb, f1_nb = train_predict(model=nb_classifier, x_train=x_train,
                                 y_train=y_train, x_test=x_val, y_test=y_val,
                                 c_str='Naive Bayes')
    f1.append(f1_nb)
    c_str.append('Naive Bayes')

    # **LOGISTIC REGRESSION**
    # L2 regularization:
    for c in [0.01, 0.1, 1, 10]:
        s = 'Logistic Regression, L2, C={}'.format(c)
        lr2_classifier = LogisticRegression(random_state=0, solver='lbfgs',
                                            penalty='l2', max_iter=1000,
                                            C=c)
        cm_lr2, f1_lr2 = train_predict(model=lr2_classifier, x_train=x_train,
                                       y_train=y_train, x_test=x_val,
                                       y_test=y_val,
                                       c_str=s)
        f1.append(f1_lr2)
        c_str.append(s)

    # L1 regularization:
    for c in [0.01, 0.1, 1, 10]:
        s = 'Logistic Regression, L1, C={}'.format(c)
        lr1_classifier = LogisticRegression(random_state=0, solver='saga',
                                            penalty='l1', max_iter=1000,
                                            C=c)
        cm_lr1, f1_lr1 = train_predict(model=lr1_classifier,
                                       x_train=x_train,
                                       y_train=y_train, x_test=x_val,
                                       y_test=y_val,
                                       c_str=s)
        f1.append(f1_lr1)
        c_str.append(s)

    # ** Stochastic Gradient Descent **
    # L2 regularization:
    sgd_classifier2 = SGDClassifier(max_iter=1000, tol=1e-4, penalty='l2')
    # noinspection PyTypeChecker
    cm_sgd2, f1_sgd2 = train_predict(model=sgd_classifier2, x_train=x_train,
                                     y_train=y_train, x_test=x_val,
                                     y_test=y_val,
                                     c_str='Stochastic Gradient Descent, L2')
    f1.append(f1_sgd2)
    c_str.append('Stochastic Gradient Descent, L2')

    # L1 regularization:
    for a in [0.0001, 0.001, 0.01, 0.1]:
        s = 'Stochastic Gradient Descent, L1, alpha={}'.format(a)
        sgd_classifier1 = SGDClassifier(max_iter=1000, tol=1e-4, penalty='l1',
                                        alpha=a)
        # noinspection PyTypeChecker
        cm_sgd1, f1_sgd1 = \
            train_predict(model=sgd_classifier1,
                          x_train=x_train,
                          y_train=y_train, x_test=x_val,
                          y_test=y_val,
                          c_str=s)
        f1.append(f1_sgd1)
        c_str.append(s)

    # L2, modified_huber loss
    sgd_classifier_mh = SGDClassifier(max_iter=1000, tol=1e-4, penalty='l2',
                                      loss='modified_huber')
    # noinspection PyTypeChecker
    cm_sgd_mh, f1_sgd_mh = \
        train_predict(model=sgd_classifier_mh,
                      x_train=x_train,
                      y_train=y_train, x_test=x_val,
                      y_test=y_val,
                      c_str='Stochastic Gradient Descent, MH')
    f1.append(f1_sgd_mh)
    c_str.append('Stochastic Gradient Descent, MH')

    # L2, squared_hinge loss
    sgd_classifier_sh = SGDClassifier(max_iter=1000, tol=1e-4, penalty='l2',
                                      loss='squared_hinge')
    # noinspection PyTypeChecker
    cm_sgd_sh, f1_sgd_sh = \
        train_predict(model=sgd_classifier_sh,
                      x_train=x_train,
                      y_train=y_train, x_test=x_val,
                      y_test=y_val,
                      c_str='Stochastic Gradient Descent, SH')
    f1.append(f1_sgd_sh)
    c_str.append('Stochastic Gradient Descent, SH')

    ####################################################################
    # Determine best model. Hard-code retrain and test on testing data.
    ####################################################################
    print('Best F1 score: {:.4f}'.format(max(f1)))
    # noinspection PyTypeChecker
    print('Corresponding model: {}'.format(c_str[np.argmax(np.array(f1))]))

    sgd_classifier1 = SGDClassifier(max_iter=1000, tol=1e-4, penalty='l1',
                                    alpha=0.001)
    # noinspection PyTypeChecker
    cm_sgd1, f1_sgd1 = \
        train_predict(model=sgd_classifier1,
                      x_train=np.vstack((x_train, x_val)),
                      y_train=np.hstack((y_train, y_val)),
                      x_test=x_test, y_test=y_test,
                      c_str='Stochastic Gradient Descent, L1, alpha=0.001')

    ####################################################################
    # Step 8 (labeled step 6 in the prompt)
    ####################################################################
    # Create RandomForest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # print feature importances in descending order
    for f in range(x_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f],
                                       importances[indices[f]]))

    # Extract top two features from our already scaled data.
    x_train_2 = x_train[:, indices[0:2]]
    x_val_2 = x_val[:, indices[0:2]]
    x_test_2 = x_test[:, indices[0:2]]

    # Perform Naive Bayes again.
    # Initialize and train.
    nb2 = GaussianNB()
    # Train, predict, and score. Use the full training + validation set
    # for training, and the testing dataset for testing.
    # noinspection PyTypeChecker
    cm_nb2, f1_nb2 = train_predict(model=nb2,
                                   x_train=np.vstack((x_train_2, x_val_2)),
                                   y_train=np.hstack((y_train, y_val)),
                                   x_test=x_test_2,
                                   y_test=y_test,
                                   c_str='Naive Bayes, 2 Features')

    # Plot the two most important features on a scatter.
    df_2 = pd.DataFrame(dict(x1=x_val_2[:, 0], x2=x_val_2[:, 1], label=y_val))
    colors = {1: 'blue', 0: 'red'}
    fig, ax = plt.subplots()
    grouped = df_2.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x1', y='x2', marker='o', s=100,
                   alpha=0.2, label=key,
                   color=colors[key])

    plt.show()

    # Set min and max values and give it some padding
    x_min, x_max = x_val_2[:, 0].min() - .5, x_val_2[:, 0].max() + .5
    y_min, y_max = x_val_2[:, 1].min() - .5, x_val_2[:, 1].max() + .5
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    z = nb2.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    # Plot the contour and test examples
    # noinspection PyUnresolvedReferences
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    # noinspection PyUnresolvedReferences
    plt.scatter(x_val_2[:, 0], x_val_2[:, 1], c=y_val, cmap=plt.cm.Spectral)

    plt.show()

    pass


if __name__ == '__main__':
    main()
