import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

def run_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Create visualization
    fig = px.scatter(x=y_test, y=y_pred, 
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Gradient Boosting: Actual vs Predicted")
    fig.add_shape(type="line", line=dict(dash='dash'),
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max())
    
    return model, mse, fig

def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    fig = px.scatter(x=y_test, y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Linear Regression: Actual vs Predicted")
    fig.add_shape(type="line", line=dict(dash='dash'),
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max())
    
    return model, mse, fig

def run_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    fig = px.scatter(x=y_test, y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Random Forest: Actual vs Predicted")
    fig.add_shape(type="line", line=dict(dash='dash'),
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max())
    
    return model, mse, fig