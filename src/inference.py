import pandas as pd
import mlflow.pyfunc

def run_local_inference():
    # Load the model directly from the Model Registry using the Production tag
    model_name = "ChurnPredictor"
    stage = "Production"
    model_uri = f"models:/{model_name}/{stage}"
    
    print(f"Loading model from: {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Simple dummy payload matching your training dimensions
    sample_data = pd.DataFrame({
        'col1': [2.5],
        'col2': [1.8]
    })
    
    predictions = model.predict(sample_data)
    print(f"Local Inference Prediction output: {predictions}")

if __name__ == "__main__":
    run_local_inference()
