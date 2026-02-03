# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,ConfusionMatrixDisplay, classification_report)

df = pd.read_csv("data/features/features.csv")

# Columns that leak outcome and MUST be excluded as they may lead to accuracy of 100% which is not desirable
leakage_cols = [ 'revenue', 'profit', 'roi','vote_count', 'vote_average','title', 'release_date', 'id']

# Features: Start with all columns and remove/drop leakage columns + target
X_clf = df.drop(columns = leakage_cols + ['hitflop'])

# Target variable
y_clf = df['hitflop']

#Train test Split
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

def save_confusion_matrix(y_test, preds):

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(values_format="d")

    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    
#Model to display performance metrics
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Train the model, predict, and print evaluation metrics.
    Returns a dict of metrics for comparison.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # some models may not have predict_proba
        y_proba = None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }

#Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000,random_state=42)

metrics_log = evaluate_model("Logistic Regression", log_reg, X_train, X_test, y_train,y_test)

#Model 2: Random Forest 
rf_clf = RandomForestClassifier( n_estimators=100, random_state=42, n_jobs=-1)

metrics_rf = evaluate_model("Random Forest", rf_clf, X_train, X_test, y_train, y_test)

#Model 3: Gradient Boosting
gb_clf = GradientBoostingClassifier(random_state=42)

metrics_gb = evaluate_model( "Gradient Boosting", gb_clf, X_train, X_test, y_train, y_test)

#Comparing the three model metrics
metrics_list = [metrics_log, metrics_rf, metrics_gb]
results_df = pd.DataFrame(metrics_list).set_index("model")
print(results_df)

#Function to evaluate the train vs test metrics of Random Forest and Gradient Boosting
def train_test_summary(name, model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)
    train_f1  = f1_score(y_train, y_train_pred)
    test_f1   = f1_score(y_test, y_test_pred)
    
    print(f"\n=== {name} Train vs Test ===")
    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Train F1      : {train_f1:.4f}, Test F1      : {test_f1:.4f}")
    print(f"Accuracy Gap  : {train_acc - test_acc:.4f}")
    print(f"F1 Gap        : {train_f1 - test_f1:.4f}")
    
    return {
        "model": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1
    }

rf_fit_summary = train_test_summary("Random Forest", rf_clf, X_train, X_test, y_train, y_test)
gb_fit_summary = train_test_summary("Gradient Boosting", gb_clf, X_train, X_test, y_train, y_test)

def plot_feature_importance(model, X):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_n = 15  # show top 15 only

    plt.figure(figsize=(10,6))
    plt.barh(
        range(top_n),
        importances[indices][:top_n][::-1]
    )
    plt.yticks(
        range(top_n),
        X.columns[indices][:top_n][::-1]
    )

    plt.xlabel("Importance")
    plt.title("Top Feature Importance")

    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()

best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

plot_feature_importance(best_model, X_train)



