import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/features/features.csv")

def tune():
    #Identifying leakage cols as them may provide highest accuracy which is not desirable
    reg_leakage_cols = ['revenue','profit','roi','hitflop','title','release_date','id','vote_count','vote_average']

    # Feature matrix X for regression: everything except leakage + target
    X_reg = df.drop(columns=reg_leakage_cols)

    # Target vector y
    y_reg = df['roi']

    #Train Test Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    #----Tuning Random Forest Regressor-----
    rf_reg_base = RandomForestRegressor(random_state=42)

    param_grid_rf_reg = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    grid_rf_reg = GridSearchCV(
        estimator=rf_reg_base,
        param_grid=param_grid_rf_reg,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_rf_reg.fit(X_train_reg, y_train_reg)

    best_rf_reg = grid_rf_reg.best_estimator_
    print("Best RF Regressor params:", grid_rf_reg.best_params_)
    print("Best RF CV RMSE:", -grid_rf_reg.best_score_)

    # Evaluate tuned RF on test set
    y_pred_rf_test = best_rf_reg.predict(X_test_reg)
    rf_test_r2   = r2_score(y_test_reg, y_pred_rf_test)
    rf_test_rmse = mean_squared_error(y_test_reg, y_pred_rf_test, squared=False)
    rf_test_mae  = mean_absolute_error(y_test_reg, y_pred_rf_test)

    print(f"RF Test R2   : {rf_test_r2:.4f}")
    print(f"RF Test RMSE : {rf_test_rmse:.4f}")
    print(f"RF Test MAE  : {rf_test_mae:.4f}")

    # ---- Gradient Boosting Tuning ----
    gb_reg_base = GradientBoostingRegressor(random_state=42)

    param_grid_gb_reg = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.7, 1.0]
    }

    grid_gb_reg = GridSearchCV(
        estimator=gb_reg_base,
        param_grid=param_grid_gb_reg,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_gb_reg.fit(X_train_reg, y_train_reg)

    best_gb_reg = grid_gb_reg.best_estimator_
    print("Best GB Regressor params:", grid_gb_reg.best_params_)
    print("Best GB CV RMSE:", -grid_gb_reg.best_score_)

    # Evaluate tuned Gradient Boosting on test set
    y_pred_gb_test = best_gb_reg.predict(X_test_reg)
    gb_test_r2   = r2_score(y_test_reg, y_pred_gb_test)
    gb_test_rmse = mean_squared_error(y_test_reg, y_pred_gb_test, squared=False)
    gb_test_mae  = mean_absolute_error(y_test_reg, y_pred_gb_test)

    print(f"GB Test R2   : {gb_test_r2:.4f}")
    print(f"GB Test RMSE : {gb_test_rmse:.4f}")
    print(f"GB Test MAE  : {gb_test_mae:.4f}")

if __name__ == "__main__":
    tune()