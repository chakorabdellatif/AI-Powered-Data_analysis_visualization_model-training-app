from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import numpy as np

def run_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=np.unique(y_test),
                    y=np.unique(y_test),
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    return model, acc, fig

def run_random_forest_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=np.unique(y_test),
                    y=np.unique(y_test),
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    return model, acc, fig

def run_gradient_boosting_classifier(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=np.unique(y_test),
                    y=np.unique(y_test),
                    text_auto=True)
    fig.update_layout(title="Confusion Matrix")
    return model, acc, fig