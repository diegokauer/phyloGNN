import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV

from phylo_gnn.data_factory.transformations import (
    apply_presence_absence,
    apply_relative_abundance,
    apply_arcsine_square_root,
    apply_centered_log_ratio,
    apply_robust_centered_log_ratio,
    apply_additive_log_ratio,
    apply_isometric_log_ratio
)
from phylo_gnn.data_factory.tabular_factory import TabularDataFactory

np.random.seed(42)

dataset = TabularDataFactory()
train = dataset.get_split('train')
test = dataset.get_split('test')
composition_cols = dataset.composition_columns
patient_metadata_cols = dataset.metadata_columns
target_col = ['Deterioration']

kwargs = {
    'use_pseudo_counts': True,
    'c': 1,
    'columns': composition_cols,
    'reference_columns': ['b_Pseudomonas'] # Pseudomonas
}

functions = [
    apply_presence_absence,
    apply_relative_abundance,
    apply_arcsine_square_root,
    apply_centered_log_ratio,
    apply_robust_centered_log_ratio,
    apply_isometric_log_ratio,
    apply_additive_log_ratio,
]
names = [
    'presence_absence',
    'relative_abundance',
    'arcSine_square_root',
    'CLR',
    'rCLR',
    'ILR',
    'ALR'
]

param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'n_jobs': [10]
}

print(train[['Sex_M', 'Col_Pyo_Yes']].head())

# Function to find best threshold for F1 score
def best_f1_threshold(y_true, y_proba):
    _, _, thresholds = roc_curve(y_true, y_proba)
    f1_scores = [f1_score(y_true, y_proba >= t) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

for fn, name in zip(functions, names):
    print(name)

    train_dataset = train[['id_patient', 'split'] + target_col + patient_metadata_cols + composition_cols]
    train_dataset = fn(train_dataset, **kwargs)

    test_dataset = test[['id_patient', 'split'] + target_col + patient_metadata_cols + composition_cols]
    test_dataset = fn(test_dataset, **kwargs)

    original_dataset = pd.concat([train_dataset, test_dataset], axis=0)
    original_dataset.to_csv(f"../../data/{name}_transformed.csv", index=False)


    cols = [col for col in patient_metadata_cols + composition_cols if col in train_dataset.columns]
    X_train = train_dataset[cols]
    y_train = train_dataset[target_col]
    X_test = test_dataset[cols]
    y_test = test_dataset[target_col]

    # Grid search with cross-validation
    # clf = RandomForestClassifier(random_state=42, oob_score=roc_auc_score)
    clf = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=[1, 10], max_iter=1000)
    # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    # # grid_search.fit(X_train, y_train.to_numpy().ravel())
    clf.fit(X_train, y_train.to_numpy().ravel())
    # print(clf.intercept_)
    # for i in zip(clf.coef_, clf.classes_):
    #     print(i)

    # best_clf = grid_search.best_estimator_
    # print(f"Best parameters: {grid_search.best_params_}")
    best_clf = clf

    # Evaluate the best model
    y_prob = best_clf.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = best_f1_threshold(y_test.to_numpy().ravel(), y_prob)
    y_pred = (y_prob > best_threshold).astype(int)

    precision, recall, thrsh = precision_recall_curve(y_true=y_test, probas_pred=clf.predict_proba(X_test)[:, 1])

    # Report results
    # print(clf.score(X_test, y_test))
    print(f"Train ROC-AUC: {roc_auc_score(y_true=y_train, y_score=best_clf.predict_proba(X_train)[:,1]):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_true=y_test, y_score=y_prob):.4f}")
    print(f"Test PR-AUC: {auc(recall, precision):.4f}")
    print(f"Best Test f1-Score: {best_f1:.4f} at Threshold: {best_threshold:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true=y_test, y_pred=y_pred)}")

    # print(f"Logistic Regression coeficients: {}")


    # Feature importance
    # importances = pd.Series(best_clf.feature_importances_, index=best_clf.feature_names_in_).sort_values(
    #     ascending=False)
    # print(f"5 Most important cols: \n{importances[:5]}\n")
