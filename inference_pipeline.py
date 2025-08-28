from typing import Annotated
import pandas as pd

from zenml import step, pipeline, Model, get_step_context
from zenml.config import DockerSettings
from zenml.enums import ModelStages


@step
def inference(model_artifact: str = "price_prediction_model") -> Annotated[pd.DataFrame, "inference_predictions"]:
    """Load model and run inference on a single synthetic data sample."""
    zenml_model = get_step_context().model
    
    # Load the trained model
    model = (
        zenml_model.get_model_artifact(model_artifact)
                   .load()             
        ) 
    
    print(f"Loaded model: {type(model)}")
    
    # Create a single row of synthetic data ad-hoc
    sample_data = {
        'category': 'Electronics',
        'discount_offered': True,
        'brand_rating': 4.2,
        'num_reviews': 150,
        'days_since_release': 45,
        'shipping_weight': 2.1,
        'competitors_price': 199.99,
        'manufacturing_cost': 85.50,
        'tax_rate': 0.08
    }
    
    # Convert to DataFrame
    X_inference = pd.DataFrame([sample_data])
    
    print("Sample input data:")
    print(X_inference.to_string(index=False))
    
    # Run inference
    predictions = model.predict(X_inference)
    
    # Create results DataFrame with input features and predictions
    results_df = X_inference.copy()
    results_df['predicted_price'] = predictions[0]
    
    return results_df


@pipeline(
    enable_cache=False,
    model=Model(
        name="PricePredictionModel",
        description="Show case Model Control Plane.",
        version=ModelStages.LATEST  # Replace with ModelStages.PRODUCTION to attach to the production model
    ),
    settings={
        "docker": DockerSettings(  # Only relevant when you start orchestrating on Docker
            requirements="requirements.txt",
            python_package_installer="uv",
        ),
    },
)
def price_prediction_inference():
    inference()

if __name__ == "__main__":
    price_prediction_inference()