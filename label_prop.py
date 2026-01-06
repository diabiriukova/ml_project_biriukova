from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading

def impute(column):
    missing = column.isnull()
    filled = resample(column[~missing], n_samples=missing.sum(), replace=True)
    column[missing] = filled
    return column

secondary_mushroom = fetch_ucirepo(id=848)
X = secondary_mushroom.data.features
y = secondary_mushroom.data.targets

#Drop where there are too many null values
X.drop(columns=['gill-spacing','stem-surface','stem-root', 'spore-print-color', 'veil-type', 'veil-color'],axis=1,inplace=True)


label_encoder = LabelEncoder()

#Impute where there are some null values
X['cap-surface-encoded'] = label_encoder.fit_transform(X['cap-surface'])
cap_surface_data = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
X['cap-surface-encoded'].replace({11: np.nan},inplace=True)
X['cap-surface-imputed'] = impute(X['cap-surface-encoded'])
cap_surface_data = {value: key for key, value in cap_surface_data.items()}
X['cap-surface-imputed'] = X['cap-surface-imputed'].map(cap_surface_data)



X['gill-attachment-encoded'] = label_encoder.fit_transform(X['gill-attachment'])
gill_attachment_data = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
X['gill-attachment-encoded'].replace({7:np.nan},inplace=True)
X['gill-attachment-imputed'] = impute(X['gill-attachment-encoded'])
gill_attachment_data = {value: key for key, value in gill_attachment_data.items()}
X['gill-attachment-imputed'] = X['gill-attachment-imputed'].map(gill_attachment_data)



X['ring-type-encoded'] = label_encoder.fit_transform(X['ring-type'])
ring_type_data = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
X['ring-type-encoded'].replace({8:np.nan},inplace=True)
X['ring-type-imputed'] = impute(X['ring-type-encoded'])
ring_type_data = {value: key for key, value in ring_type_data.items()}
X['ring-type-imputed'] = X['ring-type-imputed'].map(ring_type_data)



X_imputed = X.drop(columns=['ring-type', 'gill-attachment', 'cap-surface' , 'cap-surface-encoded', 'ring-type-encoded', 'gill-attachment-encoded'],axis=1)



#LABEL ENCODING + One-Hot
y = label_encoder.fit_transform(y)
class_encoded_data = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

X_dummies = pd.get_dummies(X_imputed,dtype=int,drop_first=True)
#print(X_dummies)

X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=42)


#Take only 100 labeled instances
np.random.seed(42)

n_labeled = 1000
indices = np.random.permutation(len(X_train))

labeled_idx = indices[:n_labeled]
unlabeled_idx = indices[n_labeled:]

X_labeled = X_train.iloc[labeled_idx]
y_labeled = y_train[labeled_idx]

X_unlabeled = X_train.iloc[unlabeled_idx]


print("Split the dataset to labeled and unlabeled")
print("All samples: ", X_train.shape)
print("Labeled samples:", X_labeled.shape[0])
print("Labeled samples y :", y_labeled.shape)
print("Unlabeled samples:", X_unlabeled.shape[0])
print("Test samples:", X_test.shape[0])


#PCA
n_components = 30
pca = PCA(n_components=n_components)

pca.fit(X_train)

X_train= pca.transform(X_train)
X_labeled = pca.transform(X_labeled)
X_unlabeled = pca.transform(X_unlabeled)
X_test = pca.transform(X_test)

#X_all = X_train
y_all = np.concatenate([
    y_labeled,
    -1 * np.ones(len(X_unlabeled))  # -1 = unknown label
])

label_model = LabelSpreading(
    kernel='knn',
    n_neighbors=10,
    alpha=0.8,
    max_iter=60)

label_model.fit(X_train, y_all)

# Final labels after propagation
y_unlabeled_pred = label_model.transduction_[-len(X_unlabeled):]

print(f"Predicted labels for {len(y_unlabeled_pred)} unlabeled samples")

y_unlabeled_probs = label_model.label_distributions_[-len(X_unlabeled):]

confidence = np.max(y_unlabeled_probs, axis=1)
mask = confidence > 0.9

X_pseudo = X_unlabeled[mask]
y_pseudo = y_unlabeled_pred[mask]

print(f"{len(X_pseudo)} high-confidence pseudo-labels selected")

X_train_rf = np.vstack([X_labeled, X_pseudo])
y_train_rf = np.concatenate([y_labeled, y_pseudo])

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

rf.fit(X_train_rf, y_train_rf)
y_pred_train_rf = rf.predict(X_train_rf)
y_pred_test_rf = rf.predict(X_test)
accuracy_train = accuracy_score(y_train_rf, y_pred_train_rf)
accuracy_test = accuracy_score(y_test, y_pred_test_rf)
print('Random Forest with Label Propagation :')
print('Train accuracy:')
print(f'{accuracy_train}\n')

print('Test accuracy:')
print(f'{accuracy_test}\n')