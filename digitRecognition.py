import os.path
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.filters import sobel
from sklearn.datasets import load_digits, fetch_openml
from sklearn.base import BaseEstimator, TransformerMixin
# Pour utiliser un autre dataset, par exemple MNIST :

# Chargement des données

if os.path.isfile("test_data.npy"):
    X_test = np.load("test_data.npy")
    y_test = np.load("test_labels.npy")
    X_train, y_train = load_digits(return_X_y=True)
    
else:
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110625)
# mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# X, y = mnist.data[:1000], mnist.target[:1000].astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=110625)


# Fonction de zonage
def extract_zone_features(X):
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
    features = np.stack([
        upper.mean(axis=(1, 2)),
        middle.mean(axis=(1, 2)),
        lower.mean(axis=(1, 2))
    ], axis=1)
    return features

# Classes de transformation
class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        img_size = int(np.sqrt(X.shape[1]))
        return np.array([np.mean(sobel(img.reshape((img_size, img_size)))) for img in X]).reshape(-1, 1)

class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return extract_zone_features(X)

# Pipeline
all_features = FeatureUnion([
    ('pca', PCA(n_components=20)),
    ('zones', ZonalInfoPreprocessing()),
    ('sobel', EdgeInfoPreprocessing())
])

clf = Pipeline([
    ('prescale', MinMaxScaler()),
    ('features', all_features),
    ('postscale', StandardScaler()),
    ('classifier', SVC(kernel='linear', C=1.0))
])

# GridSearch
param_grid1 = {
    'features__pca__n_components': [15, 20, 25],
    'prescale': [MinMaxScaler(), StandardScaler()],
    'postscale': [MinMaxScaler(), StandardScaler()],
    'classifier__C': [0.1, 1.0, 10.0],
}
clfrbf = Pipeline([
    ('prescale', MinMaxScaler()),
    ('features', all_features),
    ('postscale', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='0.1'))
])
param_grid2 = {
    'features__pca__n_components': [20, 25],
    'prescale': [MinMaxScaler(), StandardScaler()],
    'postscale': [MinMaxScaler(), StandardScaler()],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__gamma': [0.01,0.1,1.0]
}

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


# Évaluation finale

print(f"Score on the test set for linear {best_clf.score(X_test, y_test)}")
print(f"Score on the test set for rbf {best_clfrbf.score(X_test, y_test)}")
