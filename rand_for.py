from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

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


lab_enc = LabelEncoder()

#Impute where there are some null values
X['cap-surface-encoded'] = lab_enc.fit_transform(X['cap-surface'])
cap_surface_data = dict(zip(lab_enc.classes_, range(len(lab_enc.classes_))))
X['cap-surface-encoded'].replace({11: np.nan},inplace=True)
X['cap-surface-imputed'] = impute(X['cap-surface-encoded'])
cap_surface_data = {value: key for key, value in cap_surface_data.items()}
X['cap-surface-imputed'] = X['cap-surface-imputed'].map(cap_surface_data)



X['gill-attachment-encoded'] = lab_enc.fit_transform(X['gill-attachment'])
gill_attachment_data = dict(zip(lab_enc.classes_, range(len(lab_enc.classes_))))
X['gill-attachment-encoded'].replace({7:np.nan},inplace=True)
X['gill-attachment-imputed'] = impute(X['gill-attachment-encoded'])
gill_attachment_data = {value: key for key, value in gill_attachment_data.items()}
X['gill-attachment-imputed'] = X['gill-attachment-imputed'].map(gill_attachment_data)



X['ring-type-encoded'] = lab_enc.fit_transform(X['ring-type'])
ring_type_data = dict(zip(lab_enc.classes_, range(len(lab_enc.classes_))))
X['ring-type-encoded'].replace({8:np.nan},inplace=True)
X['ring-type-imputed'] = impute(X['ring-type-encoded'])
ring_type_data = {value: key for key, value in ring_type_data.items()}
X['ring-type-imputed'] = X['ring-type-imputed'].map(ring_type_data)


joblib.dump(cap_surface_data, "cap_surface_map.pkl")
joblib.dump(gill_attachment_data, "gill_attachment_map.pkl")
joblib.dump(ring_type_data, "ring_type_map.pkl")
X_imputed = X.drop(columns=['ring-type', 'gill-attachment', 'cap-surface' , 'cap-surface-encoded', 'ring-type-encoded', 'gill-attachment-encoded'],axis=1)


#LABEL ENCODING + One-Hot
y = lab_enc.fit_transform(y)
class_encoded_data = dict(zip(lab_enc.classes_, range(len(lab_enc.classes_))))

X_dummies = pd.get_dummies(X_imputed,dtype=int,drop_first=True)
#print(X_dummies)
dummy_columns = X_dummies.columns
joblib.dump(dummy_columns, "dummy_columns.pkl")
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=42)


#MODEL
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

#SAVE
joblib.dump(model, 'model_rf.pkl')
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print('Random Forest on labeled  42746 instances :')
print('Train accuracy:')
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f'{accuracy_train}\n')

print('Test accuracy:')
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'{accuracy_test}\n')


#Take only 100 labeled instances
np.random.seed(42)

n_labeled = 200
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
n_components = 40
pca = PCA(n_components=n_components)

pca.fit(X_train)

X_labeled = pca.transform(X_labeled)
X_unlabeled = pca.transform(X_unlabeled)
X_test = pca.transform(X_test)

model2 = RandomForestClassifier(
    n_estimators=200,       # more trees for stability
    max_depth=None,         # grow trees fully (or tune)
    class_weight='balanced',# handle class imbalance
    random_state=42
)
model2.fit(X_labeled, y_labeled)

y_pred_test2 = model2.predict(X_test)
y_pred_train2 = model2.predict(X_labeled)

print('\nRandom Forest on labeled 100 instances :')
print('Train accuracy:')
accuracy_train = accuracy_score(y_labeled, y_pred_train2)
print(f'{accuracy_train}\n')

print('Test accuracy:')
accuracy_test = accuracy_score(y_test, y_pred_test2)
print(f'{accuracy_test}\n')



#PSEUDO-LABELING
iteration = 0

training_accuracy = []
testing_accuracy = []
pseudo_labels = []

new_added = True

while new_added:
    model3 = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
    model3.fit(X_labeled, y_labeled)
    y_pred_test3 = model3.predict(X_test)
    y_pred_train3 = model3.predict(X_labeled)

    accuracy_train = accuracy_score(y_labeled, y_pred_train3)
    accuracy_test = accuracy_score(y_test, y_pred_test3)

    training_accuracy.append(accuracy_train)
    testing_accuracy.append(accuracy_test)

    probs = model3.predict_proba(X_unlabeled)
    predictions = model3.predict(X_unlabeled)

    threshold = 0.95
    mask_high_conf = (probs[:, 0] > threshold) | (probs[:, 1] > threshold)
    mask_remaining = ~mask_high_conf

    X_pseudo = X_unlabeled[mask_high_conf]  # features of confident samples
    y_pseudo = predictions[mask_high_conf]  # predicted labels
    if len(X_pseudo) > 0:
        new_added = True
    else:
        new_added = False
        break

    X_labeled = np.vstack([X_labeled, X_pseudo])
    y_labeled = np.concatenate([y_labeled, y_pseudo])

    X_unlabeled = X_unlabeled[mask_remaining]

    # Update iteration counter
    iteration += 1
