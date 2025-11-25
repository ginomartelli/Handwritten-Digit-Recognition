# Split the data 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Load processed feature matrix and labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from scipy.ndimage import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

from tensorflow import keras


# TODO: Add any util functions you may have from the previous script


# TODO: Load the raw data
digits = datasets.load_digits()
X,y = digits.data , digits.target 

#####
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TODO: Print dataset split summary...
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)
plt.figure()
width = 0.4
classes = np.arange(10)
train_counts = np.bincount(y_train, minlength=10)
test_counts = np.bincount(y_test, minlength=10)
plt.bar(classes - width/2, train_counts, width=width, label='Entraînement')
plt.bar(classes + width/2, test_counts, width=width, label='Test')
plt.xlabel('Classe du chiffre')
plt.ylabel('Nombre d\'exemples')
plt.title('Distribution des classes dans les ensembles d\'entraînement et de test')
plt.xticks(classes)
plt.legend()
plt.show()


# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.


##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
# 
# The PCA shall not be fitted using the test set. 
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and 
# edge features. 
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
# 
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method. 
#
# Check this documentation to understand how to work with these things 
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix
from sklearn.base import BaseEstimator, TransformerMixin
import time

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature

# TODO: Fill out the useful code for this class
class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self# No fitting needed for this processing
    
    def transform(self, X):
        n_features = X.shape[1]
        img_size = int(np.sqrt(n_features))
        if img_size * img_size != n_features:
            raise ValueError("Images are not square or have unexpected size.")
        # Indices pour 3 zones horizontales
        zone_height = img_size // 3
        reshaped = X.reshape((-1, img_size, img_size))
        upper = reshaped[:, :zone_height, :]
        middle = reshaped[:, zone_height:2 * zone_height, :]
        lower = reshaped[:, 2 * zone_height:, :]
        features = np.stack([upper.mean(axis=(1, 2)),middle.mean(axis=(1, 2)),lower.mean(axis=(1, 2))], axis=1)
        return features

# TODO: Create a single sklearn object handling the computation of all features in parallel
all_features = FeatureUnion([
    ('pca', PCA(n_components=20)),
    ('zones', ZonalInfoPreprocessing()),
    ('sobel', EdgeInfoPreprocessing())
])
F = all_features.fit(X_train,y).transform(X_train)
# Let's make sure we have the number of dimensions that we expect!
print("Nb features computed: ", F.shape[1])

# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to 
# avoid forgetting a scaling, or a feature, or ...
# 
# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
clf = Pipeline([
    ('prescale', MinMaxScaler()),
    ('features', all_features),
    ('postscale', StandardScaler()),
    ('classifier', SVC(kernel='linear', C=1.0))
])
clfrbf = Pipeline([
    ('prescale', MinMaxScaler()),
    ('features', all_features),
    ('postscale', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='auto'))
])

##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline
clf.fit(X_train, y_train)

# # TODO: Predict the outcome of the learned algorithm on the train set and then on the test set 
predict_test = clf.predict(X_test)
predict_train = clf.predict(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))


clfrbf.fit(X_train, y_train)

# # TODO: Predict the outcome of the learned algorithm on the train set and then on the test set 
predict_test = clfrbf.predict(X_test)
predict_train = clfrbf.predict(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))
# TODO: Look at confusion matrices from sklearn.metrics and 
# 1. Display a print of it
cm = confusion_matrix(y_test, predict_test)
print(cm)

# 2. Display a nice figure of it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Purples, values_format='d')
plt.title("Confusion Matrix for SVC on Test Set")
plt.show()
# 3. Report on how you understand the results


# TODO: Work out the following questions (you may also use the score function from the classifier)


##########################################
## Hyper parameter tuning and CV
##########################################
# TODO: Change from the linear classifier to an rbf kernel
# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested

# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters

clf = Pipeline([('prescale', MinMaxScaler()),('features', all_features),('postscale', StandardScaler()),('classifier', SVC(kernel='linear',C=1.0))])
clfrbf = Pipeline([('prescale', MinMaxScaler()),('features', all_features),('postscale', StandardScaler()),('classifier', SVC(kernel='rbf', C=1.0, gamma='0.1'))
])

param_grid1 = {
    'features__pca__n_components': [15, 20, 25],
    'prescale': [MinMaxScaler(), StandardScaler()],
    'postscale': [MinMaxScaler(), StandardScaler()],
    'classifier__C': [0.1, 1.0, 10.0],
}
param_grid2 = {
    'features__pca__n_components': [20, 25],
    'prescale': [MinMaxScaler(), StandardScaler()],
    'postscale': [MinMaxScaler(), StandardScaler()],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__gamma': [0.01,0.1,1.0]
}
'''
grid_search = GridSearchCV(clf, param_grid=param_grid1, cv=5, verbose=10)
grid_search.fit(X_train, y_train)



grid_searchrbf = GridSearchCV(clfrbf, param_grid=param_grid2, cv=5, verbose=10)
grid_searchrbf.fit(X_train, y_train)
print("Best parameters found for linear: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
best_clf = grid_search.best_estimator_


print("Best parameters found for RBF kernel: ", grid_searchrbf.best_params_)
print("Best cross-validation score for RBF kernel: ", grid_searchrbf.best_score_)
best_clfrbf = grid_searchrbf.best_estimator_
'''
# TODO: fit the grid search CV and 
# 1. Check the results
# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
# 3. Retrain on the whol train set, and evaluate on the test set
# 4. Answer the questions below and report on your findings

cv_values = [2, 5, 10, 15]
best_accuracies = []

param_gridcv = {
  'features__pca__n_components': [20, 25],
  'prescale': [MinMaxScaler(), StandardScaler()],
  'postscale': [MinMaxScaler(), StandardScaler()],
  'classifier__kernel': ['linear','rbf'],
  'classifier__C': [0.1, 1.0, 10.0],
}
for cv in cv_values:
  grid_search = GridSearchCV(clf, param_grid=param_gridcv, cv=cv, verbose=10)
  grid_search.fit(X_train, y_train)
  best_accuracies.append(grid_search.best_score_)

plt.figure(figsize=(7,4))
plt.plot(cv_values, best_accuracies, marker='o')
plt.xlabel('Nombre de folds (K)')
plt.ylabel('Accuracy moyenne (CV)')
plt.title('Accuracy moyenne des meilleurs paramètres en fonction de K')
plt.grid(True)
plt.show()

# #####
# print("\n Question: What happens if we change K from 5 to 10?")
# print("Test different K values and compare the accuracy variation.\n")


##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy
best_param={'classifier__C': 10.0, 'classifier__kernel': 'rbf', 'features__pca__n_components': 25, 'postscale': StandardScaler(), 'prescale': MinMaxScaler()}
clf.set_params(**best_param)

# One-vs-One (OvO)
start_ovo = time.time()
clf_ovo = OneVsOneClassifier(clf)
clf_ovo.fit(X_train, y_train)
ovo_time = time.time() - start_ovo

# One-vs-Rest (OvR)
start_ova = time.time()
clf_ova = OneVsRestClassifier(clf)
clf_ova.fit(X_train, y_train)
ova_time = time.time() - start_ova

# Print OvO results
print(" One-vs-One (OvO) Classification:")
print(f"- Test score: {clf_ovo.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf_ovo.estimators_)}")
print(f"- Training time: {ovo_time:.3f} seconds")
print("- Impact: More accurate for small datasets but can be slower with many classes.")

# Print OvR results
print(" One-vs-Rest (OvR) Classification:")
print(f"- Test score: {clf_ova.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf_ova.estimators_)}")
print(f"- Training time: {ova_time:.3f} seconds")
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")


print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")
print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")



