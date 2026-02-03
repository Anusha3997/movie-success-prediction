# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)

df = pd.read_csv("data/features/features.csv")
def tune():
    # Columns that leak outcome and MUST be excluded as they may lead to accuracy of 100% which is not desirable
    leakage_cols = [ 'revenue', 'profit', 'roi','vote_count', 'vote_average','title', 'release_date', 'id']

    # Features: Start with all columns and remove/drop leakage columns + target
    X_clf = df.drop(columns = leakage_cols + ['hitflop'])

    # Target variable
    y_clf = df['hitflop']

    #Train test Split
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # ----- Random Forest tuning -----
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid_rf = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    grid_rf = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid_rf,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    print("Best RF params:", grid_rf.best_params_)
    print("Best RF CV F1:", grid_rf.best_score_)

    # Evaluate tuned RF on test set
    from sklearn.metrics import classification_report

    y_test_pred_rf = best_rf.predict(X_test)
    print("\nTuned Random Forest - Test Classification Report:")
    print(classification_report(y_test, y_test_pred_rf, zero_division=0))

    # ----- Gradient Boosting tuning -----# ----- Gradient Boosting tuning -----
    gb_base = GradientBoostingClassifier(random_state=42)

    param_grid_gb = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 1.0],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_gb = GridSearchCV(
        estimator=gb_base,
        param_grid=param_grid_gb,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_gb.fit(X_train, y_train)

    best_gb = grid_gb.best_estimator_
    print("Best Gradient Boosting params:", grid_gb.best_params_)
    print("Best GB CV F1:", grid_gb.best_score_)

    y_test_pred_gb = best_gb.predict(X_test)
    print("\nTuned Gradient Boosting - Test Classification Report:")
    print(classification_report(y_test, y_test_pred_gb, zero_division=0))

if __name__ == "__main__":
    tune()
