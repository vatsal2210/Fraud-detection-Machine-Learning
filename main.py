# Importing Libraries
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from tensorflow.keras import losses
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import tree
import graphviz
from sklearn.utils import class_weight

print("Modules Imported! \n")

# Data preprocessing
print("--- Loading Dataset ---")
url = 'C:/Users/Kingslayer/Desktop/Project/dataset.csv'
dataset = pd.read_csv(url)

# Feature Engineering
dataset.rename(columns=lambda X: X.lower(), inplace=True)
dataset.drop('id', axis=1, inplace=True)
dataset.rename(
    columns={'default.payment.next.month': 'isDefault'}, inplace=True)

print("Dataset Info")
print("Default Credit Card Clients data -  rows:",
      dataset.shape[0], " columns:", dataset.shape[1])
dataset.describe()

# Feature Engineering
dataset['grad_school'] = (dataset['education'] == 1).astype('int')
dataset['university'] = (dataset['education'] == 2).astype('int')
dataset['high_school'] = (dataset['education'] == 3).astype('int')
dataset.drop('education', axis=1, inplace=True)

dataset['male'] = (dataset['sex'] == 1).astype('int')
dataset.drop('sex', axis=1, inplace=True)

dataset['married'] = (dataset['marriage'] == 1).astype('int')
dataset['single'] = (dataset['marriage'] == 2).astype('int')
dataset.drop('marriage', axis=1, inplace=True)

# For pay features if the <=0 then it means it was not delayed
pay_features = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
for p in pay_features:
    dataset.loc[dataset[p] <= 0, p] = 0

target_name = 'isDefault'
X = dataset.drop(target_name, axis=1)
# Robust Scaler for scaling different values into proper scale
robust_scaler = RobustScaler()
X = robust_scaler.fit_transform(X)
y = dataset[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=15, stratify=y)

# Defining a confusion matrix


def CMatrix(CM, labels=['non-defaulter', 'defaulter']):
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name = 'TRUE'
    df.columns.name = 'PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


metrics = pd.DataFrame(index=['accuracy', 'precision', 'recall'],
                       columns=['NULL', 'LogisticReg', 'ClassTree', 'NaiveBayes', 'SVM', 'KNN', 'ANN', 'VotingClassifier'])

y_pred_test = np.repeat(y_train.value_counts().idxmax(), y_test.size)

metrics.loc['accuracy', 'NULL'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'NULL'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'NULL'] = recall_score(y_pred=y_pred_test, y_true=y_test)

CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)

# A. Logistic Regression
# 1. Create an instance of the estimator
logistic_regression = LogisticRegression(n_jobs=-1, random_state=15)

# 2. Use the training data to train the estimator
logistic_regression.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = logistic_regression.predict(X_test)

metrics.loc['accuracy', 'LogisticReg'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'LogisticReg'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'LogisticReg'] = recall_score(
    y_pred=y_pred_test, y_true=y_test)

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# B. Classification Trees
# 1. Create an instance of the estimator
class_tree = DecisionTreeClassifier(
    min_samples_split=30, min_samples_leaf=10, random_state=10)

# 2. Use the training data to train the estimator
class_tree.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = class_tree.predict(X_test)

metrics.loc['accuracy', 'ClassTree'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'ClassTree'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'ClassTree'] = recall_score(
    y_pred=y_pred_test, y_true=y_test)

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# C. Naive Bayes Classifier
# 1. Create an instance of the estimator
NBC = GaussianNB()

# 2. Use the training data to train the estimator
NBC.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = NBC.predict(X_test)

metrics.loc['accuracy', 'NaiveBayes'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'NaiveBayes'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'NaiveBayes'] = recall_score(
    y_pred=y_pred_test, y_true=y_test)

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# D. -SVM
# 1. Create an instance of the estimator
svm = SVC(kernel='linear', probability=True)

# 2. Use the training data to train the estimator
svm.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = svm.predict(X_test)

metrics.loc['accuracy', 'SVM'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'SVM'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'SVM'] = recall_score(y_pred=y_pred_test, y_true=y_test)

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)


# E. - KNN
# 1. Create an instance of the estimator
KNN = KNeighborsClassifier()

# 2. Use the training data to train the estimator
KNN.fit(X_train, y_train)

# 3. Evaluate the model
y_pred_test = KNN.predict(X_test)
# metrics.loc['classification_report', 'KNN'] = classification_report(y_pred=y_pred_test, y_true=y_test)
metrics.loc['accuracy', 'KNN'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'KNN'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'KNN'] = recall_score(y_pred=y_pred_test, y_true=y_test)

# #Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)

# Data columns
x1 = dataset[['limit_bal', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5',
              'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'grad_school', 'university', 'high_school', 'male', 'married', 'single']]
y1 = dataset[['isDefault']]

encoder = LabelEncoder()
encoder.fit(y1)
encoded_Y = encoder.transform(y1)
print(encoded_Y)

print("\nSpilt Train and Test dataset")
X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(
    x1, encoded_Y, test_size=0.2)


# Artificial Neural Network
print("\nApplying Model")
ann = Sequential()
ann.add(Dense(25, input_dim=26, activation='relu'))
ann.add(Dense(20, input_dim=26, activation='relu'))
ann.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


# Compiling the model
print("\nModel Compiling...")
ann.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
print("\nModel Summery:")
print(ann.summary())
print("\nModel Training...")
history = ann.fit(X_Train, Y_Train, validation_split=0.40,
                  epochs=100, batch_size=5, verbose=0)

# Evaluating ANN
print("\nModel Evalution...")
estimator = KerasClassifier(epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

scores = ann.evaluate(X_Test, Y_Test, verbose=0)
y_pred_test = ann.predict(X_Test)
print("\n%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))

metrics.loc['accuracy', 'ANN'] = accuracy_score(
    y_pred=y_pred_test, y_true=Y_Test)
metrics.loc['precision', 'ANN'] = precision_score(
    y_pred=y_pred_test, y_true=Y_Test)
metrics.loc['recall', 'ANN'] = recall_score(y_pred=y_pred_test, y_true=Y_Test)

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=Y_Test)
CMatrix(CM)

# Ensembled Learning Technique : Voting Classifier
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(probability=True)
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train, y_train)
y_pred_test = ensemble.predict(X_test)

metrics.loc['accuracy', 'VotingClassifier'] = accuracy_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['precision', 'VotingClassifier'] = precision_score(
    y_pred=y_pred_test, y_true=y_test)
metrics.loc['recall', 'VotingClassifier'] = recall_score(
    y_pred=y_pred_test, y_true=y_test)

# #Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
CMatrix(CM)

# Scores for all the models
100*metrics

# Comparison of recall, precision and accuracy using graph
fig, ax = plt.subplots(figsize=(10, 7))
metrics.plot(kind='barh', ax=ax)
ax.grid()

precision_nb, recall_nb, thresholds_nb = precision_recall_curve(
    y_true=y_test, probas_pred=NBC.predict_proba(X_test)[:, 1])
precision_lr, recall_lr, thresholds_lr = precision_recall_curve(
    y_true=y_test, probas_pred=logistic_regression.predict_proba(X_test)[:, 1])
precision_ct, recall_ct, thresholds_ct = precision_recall_curve(
    y_true=y_test, probas_pred=class_tree.predict_proba(X_test)[:, 1])
precision_sv, recall_sv, thresholds_sv = precision_recall_curve(
    y_true=y_test, probas_pred=svm.predict_proba(X_test)[:, 1])
precision_kn, recall_kn, thresholds_kn = precision_recall_curve(
    y_true=y_test, probas_pred=KNN.predict_proba(X_test)[:, 1])
precision_ann, recall_ann, thresholds_ann = precision_recall_curve(
    y_true=y_test, probas_pred=ann.predict_proba(X_test)[:, 1])

precision_esl, recall_esl, thresholds_esl = precision_recall_curve(
    y_true=y_test, probas_pred=ensemble.predict_proba(X_test)[:, 1])

# Precision-Recall Curve
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(precision_lr, recall_lr, label='LogisticReg')
ax.plot(precision_kn, recall_kn, label='KNN')
ax.plot(precision_nb, recall_nb, label='NaiveBayes')
ax.plot(precision_ct, recall_ct, label='Classification Tree')
ax.plot(precision_sv, recall_sv, label='SVM')
ax.plot(precision_ann, recall_ann, label='ANN')
ax.plot(precision_esl, recall_esl, label='Voting Classifier')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_title('Precision-Recall Curve')

#ax.hlines(y=0.5, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid()

# Confusion matrix for modified Logistic Regression Classifier
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_lr, precision_lr[1:], label='Precision')
ax.plot(thresholds_lr, recall_lr[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Logistic Regression Classifier: Precision-Recall')

ax.hlines(y=0.48, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid()

# Confusion matrix for modified Voting Classifier
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_esl, precision_esl[1:], label='Precision')
ax.plot(thresholds_esl, recall_esl[1:], label='Recall')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Precision, Recall')
ax.set_title('Voting Classifier: Precision-Recall')

ax.hlines(y=0.42, xmin=0, xmax=1, color='red')
ax.legend()
ax.grid()

# Classifier with threshold of 0.2
y_pred_proba = logistic_regression.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba >= 0.2).astype('int')

# Confusion matrix
CM = confusion_matrix(y_pred=y_pred_test, y_true=y_test)
print("Recall: ", 100*recall_score(y_pred=y_pred_test, y_true=y_test))
print("Precision: ", 100*precision_score(y_pred=y_pred_test, y_true=y_test))
CMatrix(CM)
