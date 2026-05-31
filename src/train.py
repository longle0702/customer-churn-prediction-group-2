import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def train_model(n_estimators, max_depth):
    # Set the experiment name
    # mlflow.set_experiment("Customer_Churn_Experiment")
    
    with mlflow.start_run():
        # 1. Mock Data / Load your actual e-commerce churn dataset here
        # (Using a tiny dummy dataset just to ensure the pipeline runs)
        X = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0] * 25,
            'col2': [0.5, 1.5, 2.5, 3.5] * 25
        })
        y = [0, 1, 0, 1] * 25
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Train model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # 3. Evaluate
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # 4. Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        
        # Log the model and register it automatically under the name 'ChurnPredictor'
        mlflow.sklearn.log_model(
            sk_model=clf, 
            artifact_path="model",
            registered_model_name="ChurnPredictor"
        )
        
        print(True, f"Model trained with accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)
