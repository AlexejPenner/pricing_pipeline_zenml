import numpy as np
from typing import Annotated, Dict, Tuple, List, Union
import datetime
import click

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from zenml import log_metadata, step, pipeline, Model, ArtifactConfig, add_tags
from zenml.config import DockerSettings
from zenml.enums import ArtifactType
from zenml.materializers.materializer_registry import materializer_registry
from zenml.steps import ResourceSettings
from zenml.types import HTMLString

# Evidently imports for drift detection
from zenml.integrations.evidently.steps import evidently_report_step, EvidentlyColumnMapping
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig

from sklearn_materializer import SklearnPipelineMaterializer
from utils import mock_data, generate_data_report, generate_model_report, Country, resolve_country

materializer_registry.register_and_overwrite_type(
    key=Pipeline, 
    type_=SklearnPipelineMaterializer
)

@step
def load_reference_data(n_samples: int) -> Annotated[pd.DataFrame, "reference_data"]:
    """Load reference (baseline) synthetic product price data."""
    np.random.seed(42)  # Fixed seed for reproducible reference data
    
    df = mock_data(n_samples)
    
    log_metadata(
        artifact_name="reference_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "data_type": "reference",
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df

@step
def load_current_data(n_samples: int, simulate_drift: bool = False) -> Annotated[pd.DataFrame, "current_data"]:
    """Load current synthetic product price data with optional drift simulation."""
    np.random.seed(123)  # Different seed for current data
    
    drift_config = None
    if simulate_drift:
        # Define drift scenario: Electronics getting more expensive, Books less popular
        drift_config = {
            "price_multipliers": {
                "Electronics": 1.3,  # 30% price increase
                "Clothing": 1.1,     # 10% price increase
                "Books": 0.9         # 10% price decrease
            },
            "category_shift": {
                "Electronics": 0.35,  # Increase from 20% to 35%
                "Books": 0.1          # Decrease from 20% to 10%
            },
            "time_factor": 1.0  # Full drift effect
        }
    
    df = mock_data(n_samples, drift_config=drift_config)
    
    log_metadata(
        artifact_name="current_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "data_type": "current",
            "drift_simulated": simulate_drift,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df

@step
def load_data(n_samples: int) -> Annotated[pd.DataFrame, "raw_data"]:
    """Load synthetic product price data with various features."""
    # Create synthetic e-commerce dataset
    np.random.seed(42)
    
    df = mock_data(n_samples)
    
    # Calculate and log detailed metrics
    missing_stats = df.isnull().sum().to_dict()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    descriptive_stats = df[numeric_cols].describe().to_dict()
    # Add category-specific metrics for enhanced reporting
    category_stats = {}
    for category in df["category"].unique():
        cat_df = df[df["category"] == category]
        category_stats[category] = {
            "count": len(cat_df),
            "avg_price": float(cat_df["price"].mean()),
            "avg_cost": float(cat_df["manufacturing_cost"].mean()),
            "avg_weight": float(cat_df["shipping_weight"].dropna().mean()),
            "avg_profit_margin": float(((cat_df["price"] - cat_df["manufacturing_cost"]) / cat_df["price"]).mean())
        }
    
    categorical_stats = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_stats[col] = df[col].value_counts().to_dict()
    
    log_metadata(
        artifact_name="raw_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": missing_stats,
            "descriptive_statistics": descriptive_stats,
            "categorical_distributions": categorical_stats,
            "category_specific_stats": category_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df

@step
def analyze_data(data: pd.DataFrame) -> Annotated[Dict, "data_analysis"]:
    """Analyze the dataset and compute various statistics."""
    analysis = {}
    
    # Correlation analysis for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    analysis["correlation"] = numeric_data.corr().to_dict()
    
    # Category distribution
    analysis["category_distribution"] = data["category"].value_counts().to_dict()
    
    # Price statistics by category
    price_by_category = data.groupby("category")["price"].agg(["mean", "min", "max", "std"]).to_dict()
    analysis["price_by_category"] = price_by_category
    
    # Discount impact analysis
    discount_impact = data.groupby("discount_offered")["price"].mean().to_dict()
    analysis["discount_impact"] = discount_impact
    
    log_metadata(
        artifact_name="data_analysis",
        infer_artifact=True,
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "price_range": (float(data["price"].min()), float(data["price"].max())),
            "most_common_category": data["category"].value_counts().index[0],
            "price_correlation_factors": sorted(
                [(col, analysis["correlation"]["price"].get(col, 0)) 
                 for col in numeric_data.columns if col != "price"],
                key=lambda x: abs(x[1]),
                reverse=True
            )
        }
    )
    
    return analysis

@step
def filter_by_country(data: pd.DataFrame, country: str = "All") -> Annotated[pd.DataFrame, "filtered_data"]:
    """Filter dataset by country if specified."""
    add_tags(tags=[f"country: {country}"], infer_artifact=True)

    if country and country != "All":
        filtered_data = data[data["country"] == country].copy()
        
        log_metadata(
            artifact_name="filtered_data",
            infer_artifact=True,
            metadata={
                "country_filter": country,
                "original_rows": data.shape[0],
                "filtered_rows": filtered_data.shape[0],
                "countries_available": list(data["country"].unique()),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        return filtered_data
    else:
        log_metadata(
            artifact_name="filtered_data",
            infer_artifact=True,
            metadata={
                "country_filter": "All countries",
                "rows": data.shape[0],
                "countries_available": list(data["country"].unique()),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        return data

@step
def clean_data(data: pd.DataFrame) -> Annotated[pd.DataFrame, "cleaned_data"]:
    """Clean the dataset by handling missing values and outliers."""
    # Store pre-cleaning stats
    pre_cleaning_shape = data.shape
    pre_cleaning_nulls = data.isnull().sum().sum()
    
    # Handle missing values
    cleaned_data = data.copy()
    
    # Fill missing brand_rating with median
    cleaned_data["brand_rating"] = cleaned_data["brand_rating"].fillna(cleaned_data["brand_rating"].median())
    
    # Fill missing num_reviews with 0
    cleaned_data["num_reviews"] = cleaned_data["num_reviews"].fillna(0)
    
    # Fill missing shipping_weight with mean
    cleaned_data["shipping_weight"] = cleaned_data["shipping_weight"].fillna(cleaned_data["shipping_weight"].mean())
    
    # Handle outliers in price (capping at 3 std devs)
    mean_price = cleaned_data["price"].mean()
    std_price = cleaned_data["price"].std()
    cleaned_data["price"] = cleaned_data["price"].clip(
        lower=mean_price - 3*std_price,
        upper=mean_price + 3*std_price
    )
    
    # Log cleaning impact
    post_cleaning_shape = cleaned_data.shape
    post_cleaning_nulls = cleaned_data.isnull().sum().sum()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    post_cleaning_stats = cleaned_data[numeric_cols].describe().to_dict()
    
    log_metadata(
        artifact_name="cleaned_data",
        infer_artifact=True,
        metadata={
            "pre_cleaning_rows": pre_cleaning_shape[0],
            "pre_cleaning_missing_values": int(pre_cleaning_nulls),
            "post_cleaning_rows": post_cleaning_shape[0],
            "post_cleaning_missing_values": int(post_cleaning_nulls),
            "outliers_handled": "price",
            "descriptive_statistics": post_cleaning_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return cleaned_data


@step(
    experiment_tracker=True
)
def train_model(
    data: pd.DataFrame,
    epochs: int = 15
    ) -> Tuple[
        Annotated[Pipeline, ArtifactConfig(name="price_prediction_model", artifact_type=ArtifactType.MODEL,)], 
        Annotated[HTMLString, "model_report"]
        ]:
    """Train a model to predict product prices."""
    
    import mlflow
    mlflow.autolog()
    mlflow.set_tag("model_type", "GradientBoostingRegressor")
    mlflow.set_tag("epochs", epochs)
    mlflow.set_tag("n_samples", data.shape[0])
    mlflow.set_tag("country", data["country"].unique()[0])
    
    # Define features and target
    # Note: We exclude product_id, country, currency since they're identifiers
    categorical_features = ["category", "discount_offered"]
    numeric_features = ["brand_rating", "num_reviews", "days_since_release", 
                        "shipping_weight", "competitors_price", "manufacturing_cost", "tax_rate"]
    features = categorical_features + numeric_features
    X = data[features]
    y = data["price"]
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=epochs * 10,  # Use epochs to control complexity
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get feature importance from the model
    gbr = model.named_steps['regressor']
    
    # Get feature names after preprocessing
    feature_names = []
    # Add numeric feature names directly
    feature_names.extend(numeric_features)
    # Get one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names.extend(categorical_feature_names)
    
    # Ensure feature_importance matches the actual features
    # If the gradient boosting model has fewer features than we expect,
    # we'll just use the first N importance values
    importances = gbr.feature_importances_[:len(feature_names)] if len(gbr.feature_importances_) > len(feature_names) else gbr.feature_importances_
    
    # Map feature names to importance values
    raw_feature_importance = dict(zip(feature_names, importances))
    
    # We need to combine one-hot encoded importances back to original features
    feature_importance = {}
    for feature, importance in raw_feature_importance.items():
        # For numeric features, just copy the importance
        if feature in numeric_features:
            feature_importance[feature] = round(importance, 4)
        else:
            # For categorical features, extract the original feature name
            # Format is typically feature_value
            original_feature = feature.split('_')[0]
            feature_importance[original_feature] = round(feature_importance.get(original_feature, 0) + importance, 4)
    
    # Create learning curves by using cross-validation
    cv_scores = []
    train_sizes = [int(epochs * i/5) for i in range(1, 6)]  # 5 points on the learning curve
    
    for n in train_sizes:
        gbr_cv = GradientBoostingRegressor(n_estimators=n * 10, learning_rate=0.1, max_depth=4, random_state=42)
        model_cv = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr_cv)])
        model_cv.fit(X_train, y_train)
        train_score = -mean_squared_error(y_train, model_cv.predict(X_train))
        test_score = -mean_squared_error(y_test, model_cv.predict(X_test))
        cv_scores.append((train_score, test_score))
    
    # Extract training and validation loss
    train_loss = [-score[0] for score in cv_scores]
    val_loss = [-score[1] for score in cv_scores]
    
    # Create learning curves (real)
    learning_curve = {
        "epochs": train_sizes,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    
    # Create model output
    model_metrics = {
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
        },
        "feature_importance": feature_importance,
        "learning_curve": learning_curve,
        "model_params": {
            "epochs": epochs,
            "model_type": "GradientBoostingRegressor",
            "features": features,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }
    
    metadata = {
        "r2_score": round(float(r2), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "epochs": epochs,
        "feature_importance": feature_importance,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Log detailed metrics about the model
    log_metadata(
        artifact_name="price_prediction_model",
        infer_artifact=True,
        metadata=metadata
    )

    # Log detailed metrics about the model
    log_metadata(
        metadata=metadata
    )

    # Log detailed metrics about the model
    log_metadata(
        metadata=metadata,
        infer_model=True,
        model_name="PricePredictionModel"
    )

    return model, HTMLString(generate_model_report(data=data, model=model_metrics))

@step
def generate_data_analysis_report(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    analysis: Dict) -> Annotated[HTMLString, "data_analysis_report"]:
    """Generate an HTML report with Plotly visualizations of the data analysis."""
    
    # Log basic metadata about the report
    log_metadata(
        artifact_name="data_analysis_report",
        infer_artifact=True,
        metadata={
            "generated_at": datetime.datetime.now().isoformat(),
            "report_type": "Data Analysis Report",
            "visualizations": [
                "Price by Category", 
                "Manufacturing Cost by Category",
                "Shipping Weight by Category",
                "Profit Margin by Category",
                "Price Distribution"
            ]
        }
    )
    
    return HTMLString(generate_data_report(cleaned_data=cleaned_data, raw_data=raw_data, analysis=analysis))

# Configure the Evidently drift detection step
drift_detection_step = evidently_report_step.with_options(
    parameters=dict(
        column_mapping=EvidentlyColumnMapping(
            target="price",
            numerical_features=[
                "brand_rating", "num_reviews", "days_since_release", 
                "shipping_weight", "competitors_price", "manufacturing_cost", 
                "tax_rate", "final_price"
            ],
            categorical_features=["category", "discount_offered"],
            text_features=None,
        ),
        metrics=[
            EvidentlyMetricConfig.metric("DataQualityPreset"),
            EvidentlyMetricConfig.metric("DataDriftPreset"),
            EvidentlyMetricConfig.metric("TargetDriftPreset"),
        ],
        # Add report options for better visualization
        report_options=[
            (
                "evidently.options.ColorOptions", {
                    "primary_color": "#5a86ad",
                    "fill_color": "#fff4f2",
                    "zero_line_color": "#016795",
                    "current_data_color": "#c292a1", 
                    "reference_data_color": "#017b92",
                }
            ),
        ],
    ),
)


@pipeline
def price_prediction_training(
    n_samples: int, 
    epochs: int = 15, 
    country: Union[str, Country] = "All",
    detect_drift: bool = False,
    simulate_drift: bool = False
):
    """Pipeline that demonstrates ZenML's visualization and reporting capabilities with optional drift detection."""
    # Convert Country enum to string before passing to steps
    country_name = resolve_country(country)
    
    if detect_drift:
        # Load reference and current data for drift detection
        reference_data = load_reference_data(n_samples)
        current_data = load_current_data(n_samples, simulate_drift=simulate_drift)
        
        # Filter both datasets by country
        filtered_reference = filter_by_country(reference_data, country_name)
        filtered_current = filter_by_country(current_data, country_name)
        
        # Clean both datasets
        cleaned_reference = clean_data(filtered_reference)
        cleaned_current = clean_data(filtered_current)
        
        # Perform drift detection
        drift_report_json, drift_report_html = drift_detection_step(
            reference_dataset=cleaned_reference,
            comparison_dataset=cleaned_current
        )
        
        # Use current data for training
        raw_data = current_data
        cleaned_data = cleaned_current
        
        # Analyze current data
        data_analysis = analyze_data(raw_data)
    else:
        # Original pipeline without drift detection
        raw_data = load_data(n_samples)
        data_analysis = analyze_data(raw_data)
        filtered_data = filter_by_country(raw_data, country_name)
        cleaned_data = clean_data(filtered_data)
    
    # Train model on cleaned data
    model, model_report = train_model(cleaned_data, epochs)
    
    # Generate data analysis report
    data_report = generate_data_analysis_report(raw_data, cleaned_data, data_analysis)

    if detect_drift:
        return model, model_report, data_report, drift_report_html
    else:
        return model, model_report, data_report

@click.command()
@click.option("--config", default="training_config.yaml", help="Path to configuration file")
def main(config: str):
    """Run the price prediction training pipeline."""
    price_prediction_training.with_options(config_path=config)()
 

if __name__ == "__main__":
    main() 