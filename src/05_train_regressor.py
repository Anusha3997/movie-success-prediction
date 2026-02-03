# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/features/features.csv")

#Identifying leakage cols as them may provide highest accuracy which is not desirable
reg_leakage_cols = ['revenue','profit','roi','hitflop','title','release_date','id','vote_count','vote_average']

# Feature matrix X for regression: everything except leakage + target
X_reg = df.drop(columns=reg_leakage_cols)

# Target vector y
y_reg = df['roi']

#Train Test Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

#Function to display model metrics
def evaluate_reg_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred)
    
    # Train metrics
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    
    print(f"\n=== {name} ===")
    print(f"Train R2   : {train_r2:.4f}")
    print(f"Test  R2   : {test_r2:.4f}")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test  RMSE : {test_rmse:.4f}")
    print(f"Train MAE  : {train_mae:.4f}")
    print(f"Test  MAE  : {test_mae:.4f}")
    
    return {
        "model": name,
        "train_R2": train_r2,
        "test_R2": test_r2,
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae
    }

#Evaluating different regression models
# Model 1: Linear Regression
lin_reg = LinearRegression()

metrics_lin = evaluate_reg_model("Linear Regression", lin_reg, X_train_reg, X_test_reg, y_train_reg, y_test_reg)

#Model 2: Random Forest Regression
rf_reg  = RandomForestRegressor(random_state=42)

metrics_rf  = evaluate_reg_model("Random Forest", rf_reg, X_train_reg, X_test_reg, y_train_reg, y_test_reg)

#Model 3: Gradient Boosting Regression
gb_reg  = GradientBoostingRegressor(random_state=42)

metrics_gb  = evaluate_reg_model("Gradient Boosting", gb_reg, X_train_reg, X_test_reg, y_train_reg, y_test_reg)

#Comparing 3 Models
reg_results_df = pd.DataFrame([metrics_lin, metrics_rf, metrics_gb]).set_index("model")
print(reg_results_df)