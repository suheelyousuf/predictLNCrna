import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from Bio import SeqIO
import time

#%matplotlib inline


named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print(time_string)





rna_train = pd.read_table('../data/random_samples.txt')


def Kmers_funct(seq, size=7):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

rna_train['words'] = rna_train.apply(lambda x: Kmers_funct(x['sequence']), axis=1)


rna_train=rna_train.drop('sequence', axis=1)


rna_texts = list(rna_train['words'])
for item in range(len(rna_texts)):
    rna_texts[item] = ' '.join(rna_texts[item])
#separate labels
y_rna = rna_train.iloc[:, 0].values

cv = CountVectorizer(ngram_range=(4,4)) #The n-gram size of 4 is previously determined by testing
X = cv.fit_transform(rna_texts)
print(X.shape)

# Splitting the human dataset into the training set and test set

X_train, X_test, y_train, y_test = train_test_split(X,y_rna,test_size = 0.20)
classifier=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(7,),random_state=1)
kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(classifier, X, y_rna, cv=kfold)
print("Algorith =  Artifical Neural Networks")
print("Accuarcy: "+str(round(results.mean(),4)))




classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print("Confusion matrix for predictions on lncRNA using NV \n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
print('finish of the expriment \n with different parameters \n')

classifier=svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions on lncRNA using SVM\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

print('random forest')
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions lncRNA using RandomForest\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

print('knn')

classifier= KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions lncRNA using K NeighborsClassifier\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

print("Neural Networks")
classifier=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(6,),random_state=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Confusion matrix for predictions lncRNA using Neural Networks\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

print('AdaBoostClassifier')


seed = 7
num_trees = 50
kfold = model_selection.KFold(n_splits=10)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y_rna, cv=kfold)
print("Algorith =  AdaBoostClassifier")
print("Accuarcy: "+str(round(results.mean(),4)))
