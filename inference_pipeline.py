from typing import Annotated, Tuple
import pandas as pd

from zenml import step, pipeline, Model, get_step_context
from zenml.config import DeploymentSettings, DockerSettings
from zenml.enums import ModelStages

@step
def load_new_row(
    category: str,
    discount_offered: bool,
    brand_rating: float,
    num_reviews: int,
    days_since_release: int,
    shipping_weight: float,
    competitors_price: float,
    manufacturing_cost: float,
    tax_rate: float) -> Annotated[pd.DataFrame, "new_row"]:
    """Load a new row of data."""
    return pd.DataFrame({
        'category': [category],
        'discount_offered': [discount_offered],
        'brand_rating': [brand_rating],
        'num_reviews': [num_reviews],
        'days_since_release': [days_since_release],
        'shipping_weight': [shipping_weight],
        'competitors_price': [competitors_price],
        'manufacturing_cost': [manufacturing_cost],
        'tax_rate': [tax_rate]
    })

@step
def inference(input_data: pd.DataFrame, model_artifact: str = "price_prediction_model") -> Tuple[Annotated[pd.DataFrame, "inference_predictions"], Annotated[float, "predicted_price"]]:
    """Load model and run inference on a single synthetic data sample."""
    zenml_model = get_step_context().model
    
    # Load the trained model
    model = (
        zenml_model.get_model_artifact(model_artifact)
                   .load()             
        ) 
    
    print(f"Loaded model: {type(model)}")
    
    # Use the input DataFrame directly
    X_inference = input_data
    
    print("Sample input data:")
    print(X_inference.to_string(index=False))
    
    # Run inference
    predictions = model.predict(X_inference)
    
    # Create results DataFrame with input features and predictions
    results_df = X_inference.copy()
    results_df['predicted_price'] = predictions[0]
    
    return results_df, predictions[0]


deployment_settings = DeploymentSettings(
    app_title="Price Prediction Inference",
    app_description="Enter a row of product data and get the predicted price.",
    dashboard_files_path="web",
)

@pipeline(settings={"deployment": deployment_settings})
def price_prediction_inference(
    category: str = 'Electronics',
    discount_offered: bool = True,
    brand_rating: float = 4.2,
    num_reviews: int = 150,
    days_since_release: int = 45,
    shipping_weight: float = 2.1,
    competitors_price: float = 199.99,
    manufacturing_cost: float = 85.50,
    tax_rate: float = 0.08) -> Annotated[float, "predicted_price"]:
    new_data_point = load_new_row(category, discount_offered, brand_rating, num_reviews, days_since_release, shipping_weight, competitors_price, manufacturing_cost, tax_rate)
    _, predicted_price = inference(new_data_point)
    return predicted_price

if __name__ == "__main__":
    price_prediction_inference.with_options(config_path="config_old/inference_config.yaml")()