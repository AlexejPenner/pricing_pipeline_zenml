import numpy as np
from typing import Annotated, Dict, Tuple
import datetime
import click
import json

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from zenml import log_metadata, step, pipeline, get_step_context
from zenml.client import Client
from zenml.types import HTMLString

# Evidently imports for model performance evaluation
from zenml.integrations.evidently.steps import evidently_report_step, EvidentlyColumnMapping
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig

from utils import mock_data


@step
def load_evaluation_dataset(n_samples: int = 1000, simulate_drift: bool = False) -> Annotated[pd.DataFrame, "evaluation_data"]:
    """
    Load evaluation dataset with ground truth optimal pricing.
    
    For this example, we generate synthetic data and use 'price' as the optimal_price.
    In production, this would load from a file or database.
    
    Args:
        n_samples: Number of samples to generate
        simulate_drift: If True, simulate data drift in the evaluation dataset
    """
    np.random.seed(999)  # Different seed for evaluation data
    
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
    
    # Generate synthetic evaluation data
    df = mock_data(n_samples, drift_config=drift_config)
    
    # For evaluation, we'll use 'price' as the ground truth optimal_price
    # In a real scenario, optimal_price would come from business logic or historical optimal decisions
    df['optimal_price'] = df['price'].copy()
    
    # Add some variation to simulate that optimal_price might differ from actual price
    # This simulates that the model should predict optimal pricing, not just replicate historical prices
    optimal_price_variation = np.random.normal(1.0, 0.05, len(df))  # ±5% variation
    df['optimal_price'] = df['optimal_price'] * optimal_price_variation
    df['optimal_price'] = np.maximum(df['optimal_price'], df['manufacturing_cost'] * 1.1)  # Ensure profitability
    
    log_metadata(
        artifact_name="evaluation_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "data_type": "evaluation",
            "has_ground_truth": True,
            "ground_truth_column": "optimal_price",
            "drift_simulated": simulate_drift,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df


@step
def load_reference_data(n_samples: int) -> Annotated[pd.DataFrame, "reference_data"]:
    """Load reference (baseline) synthetic product price data without drift."""
    np.random.seed(42)  # Fixed seed for reproducible reference data (same as training)
    
    df = mock_data(n_samples)
    
    # Add optimal_price for reference data (same logic as evaluation data)
    df['optimal_price'] = df['price'].copy()
    optimal_price_variation = np.random.normal(1.0, 0.05, len(df))
    df['optimal_price'] = df['optimal_price'] * optimal_price_variation
    df['optimal_price'] = np.maximum(df['optimal_price'], df['manufacturing_cost'] * 1.1)
    
    log_metadata(
        artifact_name="reference_data",
        infer_artifact=True,
        metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "data_type": "reference",
            "drift_simulated": False,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return df


@step
def load_production_model(
    model_artifact: str = "price_prediction_model"
) -> Annotated[Pipeline, "production_model"]:
    """Load the production model from ZenML Model Control Plane."""
    zenml_model = get_step_context().model
    
    # Load the trained model from production stage
    model = (
        zenml_model.get_model_artifact(model_artifact)
                   .load()             
    )
    
    print(f"Loaded production model: {type(model)}")
    print(f"Model version: {zenml_model.version}")
    try:
        model_version = zenml_model.get_model_version(model_artifact)
        print(f"Model stage: {model_version.stage}")
    except Exception as e:
        print(f"Could not retrieve model stage: {e}")
    
    log_metadata(
        artifact_name="production_model",
        infer_artifact=True,
        metadata={
            "model_type": str(type(model)),
            "model_artifact": model_artifact,
            "model_version": zenml_model.version,
            "load_timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return model


@step
def preprocess_evaluation_data(
    evaluation_data: pd.DataFrame
) -> Annotated[pd.DataFrame, "preprocessed_evaluation_data"]:
    """Preprocess evaluation data (same preprocessing as training)."""
    # Store pre-cleaning stats
    pre_cleaning_shape = evaluation_data.shape
    pre_cleaning_nulls = evaluation_data.isnull().sum().sum()
    
    # Handle missing values (same logic as training pipeline)
    cleaned_data = evaluation_data.copy()
    
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
    
    # Handle outliers in optimal_price if it exists
    if "optimal_price" in cleaned_data.columns:
        mean_optimal = cleaned_data["optimal_price"].mean()
        std_optimal = cleaned_data["optimal_price"].std()
        cleaned_data["optimal_price"] = cleaned_data["optimal_price"].clip(
            lower=mean_optimal - 3*std_optimal,
            upper=mean_optimal + 3*std_optimal
        )
    
    # Log cleaning impact
    post_cleaning_shape = cleaned_data.shape
    post_cleaning_nulls = cleaned_data.isnull().sum().sum()
    
    log_metadata(
        artifact_name="preprocessed_evaluation_data",
        infer_artifact=True,
        metadata={
            "pre_cleaning_rows": pre_cleaning_shape[0],
            "pre_cleaning_missing_values": int(pre_cleaning_nulls),
            "post_cleaning_rows": post_cleaning_shape[0],
            "post_cleaning_missing_values": int(post_cleaning_nulls),
            "outliers_handled": ["price", "optimal_price"],
            "preprocessing_timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return cleaned_data


@step
def generate_predictions(
    preprocessed_data: pd.DataFrame,
    model: Pipeline
) -> Annotated[pd.DataFrame, "evaluation_predictions"]:
    """Generate predictions using the production model."""
    # Define features (same as training)
    categorical_features = ["category", "discount_offered"]
    numeric_features = ["brand_rating", "num_reviews", "days_since_release", 
                        "shipping_weight", "competitors_price", "manufacturing_cost", "tax_rate"]
    features = categorical_features + numeric_features
    
    # Extract features
    X_eval = preprocessed_data[features]
    
    # Generate predictions
    predictions = model.predict(X_eval)
    
    # Create results DataFrame
    results_df = preprocessed_data.copy()
    results_df['predicted_price'] = predictions
    
    # Ensure we have ground truth
    if 'optimal_price' not in results_df.columns:
        raise ValueError("Ground truth 'optimal_price' column not found in evaluation data")
    
    log_metadata(
        artifact_name="evaluation_predictions",
        infer_artifact=True,
        metadata={
            "n_predictions": len(predictions),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
            "prediction_min": float(np.min(predictions)),
            "prediction_max": float(np.max(predictions)),
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    return results_df


@step
def calculate_evaluation_metrics(
    predictions_df: pd.DataFrame
) -> Annotated[Dict, "evaluation_metrics"]:
    """Calculate comprehensive evaluation metrics."""
    y_true = predictions_df['optimal_price'].values
    y_pred = predictions_df['predicted_price'].values
    
    # Standard regression metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Category-specific metrics
    category_metrics = {}
    for category in predictions_df['category'].unique():
        cat_mask = predictions_df['category'] == category
        cat_y_true = predictions_df.loc[cat_mask, 'optimal_price'].values
        cat_y_pred = predictions_df.loc[cat_mask, 'predicted_price'].values
        
        if len(cat_y_true) > 0:
            category_metrics[category] = {
                "r2_score": float(r2_score(cat_y_true, cat_y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(cat_y_true, cat_y_pred))),
                "mae": float(mean_absolute_error(cat_y_true, cat_y_pred)),
                "mape": float(np.mean(np.abs((cat_y_true - cat_y_pred) / cat_y_true)) * 100),
                "n_samples": int(np.sum(cat_mask))
            }
    
    metrics = {
        "overall_metrics": {
            "r2_score": round(float(r2), 4),
            "mse": round(float(mse), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "mape": round(float(mape), 4)
        },
        "category_metrics": category_metrics,
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "n_samples": len(y_true)
    }
    
    log_metadata(
        artifact_name="evaluation_metrics",
        infer_artifact=True,
        metadata=metrics
    )
    
    return metrics


# Configure the Evidently model performance evaluation step
model_performance_step = evidently_report_step.with_options(
    parameters=dict(
        column_mapping=EvidentlyColumnMapping(
            target="optimal_price",
            prediction="predicted_price",
            numerical_features=[
                "brand_rating", "num_reviews", "days_since_release", 
                "shipping_weight", "competitors_price", "manufacturing_cost", 
                "tax_rate", "final_price", "price"
            ],
            categorical_features=["category", "discount_offered"],
            text_features=None,
        ),
        metrics=[
            EvidentlyMetricConfig.metric("RegressionQualityMetric"),
            EvidentlyMetricConfig.metric("DataQualityPreset"),
        ],
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


@step
def attach_metadata_to_model(
    evaluation_metrics: Dict,
    predictions_df: pd.DataFrame
) -> None:
    """Attach evaluation metadata to the production model."""
    zenml_model = get_step_context().model
    
    # Prepare comprehensive metadata
    model_metadata = {
        "evaluation_metrics": evaluation_metrics,
        "evaluation_summary": {
            "r2_score": evaluation_metrics["overall_metrics"]["r2_score"],
            "rmse": evaluation_metrics["overall_metrics"]["rmse"],
            "mae": evaluation_metrics["overall_metrics"]["mae"],
            "mape": evaluation_metrics["overall_metrics"]["mape"],
            "n_evaluation_samples": evaluation_metrics["n_samples"],
            "evaluation_date": evaluation_metrics["evaluation_timestamp"]
        },
        "model_performance": {
            "status": "evaluated",
            "performance_grade": (
                "excellent" if evaluation_metrics["overall_metrics"]["r2_score"] > 0.9 else
                "good" if evaluation_metrics["overall_metrics"]["r2_score"] > 0.7 else
                "fair" if evaluation_metrics["overall_metrics"]["r2_score"] > 0.5 else
                "poor"
            )
        },
        "category_performance": evaluation_metrics["category_metrics"],
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Attach metadata to the model
    log_metadata(
        metadata=model_metadata,
        infer_model=True,
        model_name=zenml_model.name
    )
    
    # Also log to artifact for tracking
    log_metadata(
        metadata=model_metadata
    )
    
    return 


@step
def slack_drift_alert(
    drift_report_json: str
) -> str:
    """Sends a Slack alert if drift or significant performance issues are detected."""
    alerter = Client().active_stack.alerter
    
    if alerter is None:
        print("No alerter configured in stack - skipping Slack notification")
        return "No alerter configured"
    
    step_context = get_step_context()
    pipeline_name = step_context.pipeline.name
    run_id = step_context.pipeline_run.id
    model_id = step_context.model.id
    
    run_url = f"https://cloud.zenml.io/workspaces/zenml-projects/projects/synthesia/runs/{run_id}"
    model_url = f"https://cloud.zenml.io/workspaces/zenml-projects/projects/synthesia/model-versions/{model_id}?tab=artifacts"
    
    # Parse Evidently report to check for drift/issues
    drift_detected = False
    issues = []
    
    try:
        # Parse JSON string to dict
        report_dict = json.loads(drift_report_json)
        
        # Evidently reports have a structure with metrics containing results
        # Check for various types of drift/issues
        if isinstance(report_dict, dict):
            # Check for metrics in the report
            if "metrics" in report_dict:
                for metric in report_dict["metrics"]:
                    metric_name = metric.get("metric", "")
                    result = metric.get("result", {})
                    
                    # Check for data drift
                    if "data_drift" in metric_name.lower() or "DataDrift" in metric_name:
                        drift_score = result.get("drift_score", 0)
                        if drift_score > 0.3:  # Threshold for significant drift
                            drift_detected = True
                            issues.append(f"Data drift detected (score: {drift_score:.3f})")
                    
                    # Check for target drift
                    if "target_drift" in metric_name.lower() or "TargetDrift" in metric_name:
                        drift_score = result.get("drift_score", 0)
                        if drift_score > 0.25:
                            drift_detected = True
                            issues.append(f"Target drift detected (score: {drift_score:.3f})")
                    
                    # Check for regression quality issues (model performance degradation)
                    if "regression" in metric_name.lower() or "RegressionQuality" in metric_name:
                        # Check for performance degradation metrics
                        current_r2 = result.get("current", {}).get("r2_score", None)
                        reference_r2 = result.get("reference", {}).get("r2_score", None)
                        
                        if current_r2 is not None and reference_r2 is not None:
                            r2_degradation = reference_r2 - current_r2
                            if r2_degradation > 0.1:  # R2 dropped by more than 0.1
                                drift_detected = True
                                issues.append(f"Model performance degraded: R2 dropped from {reference_r2:.3f} to {current_r2:.3f} (Δ{r2_degradation:.3f})")
                        
                        # Check RMSE increase
                        current_rmse = result.get("current", {}).get("rmse", None)
                        reference_rmse = result.get("reference", {}).get("rmse", None)
                        
                        if current_rmse is not None and reference_rmse is not None and reference_rmse > 0:
                            rmse_increase = (current_rmse - reference_rmse) / reference_rmse
                            if rmse_increase > 0.15:  # RMSE increased by more than 15%
                                drift_detected = True
                                issues.append(f"Model error increased: RMSE increased from {reference_rmse:.2f} to {current_rmse:.2f} ({rmse_increase*100:.1f}% increase)")
                        
                        # Check prediction drift
                        prediction_drift = result.get("prediction_drift", None)
                        if prediction_drift is not None and isinstance(prediction_drift, (int, float)):
                            if abs(prediction_drift) > 0.3:  # Significant prediction drift
                                drift_detected = True
                                issues.append(f"Prediction drift detected (score: {prediction_drift:.3f})")
                        
                        # Fallback: check quality score if available
                        quality_score = result.get("quality_score", None)
                        if quality_score is not None and quality_score < 0.7:
                            drift_detected = True
                            issues.append(f"Regression quality degraded (score: {quality_score:.3f})")
                    
                    # Check for data quality issues
                    if "data_quality" in metric_name.lower() or "DataQuality" in metric_name:
                        quality_status = result.get("quality_status", "ok")
                        if quality_status != "ok":
                            drift_detected = True
                            issues.append(f"Data quality issues detected: {quality_status}")
            
            # Also check at top level for drift indicators
            if "summary" in report_dict:
                summary = report_dict["summary"]
                if summary.get("drift_detected", False):
                    drift_detected = True
                    issues.append("Drift detected in summary")
            
            # Check for overall drift status
            if report_dict.get("drift_detected", False):
                drift_detected = True
                issues.append("Overall drift detected")
    except Exception as e:
        print(f"Error parsing drift report: {e}")
        # If we can't parse, don't send alert
        return f"Error parsing report: {str(e)}"
    
    if drift_detected:
        issues_text = "\n".join([f"  • {issue}" for issue in issues]) if issues else "  • Drift detected"
        
        message = f"""⚠️ **Drift Detected in Evaluation Pipeline!**

Pipeline: {pipeline_name}
Run ID: {run_id}

**Issues Found:**
{issues_text}

RUN URL: {run_url}
MODEL URL: {model_url}

Please review the drift report and consider retraining the model."""
        
        thread_id = alerter.post(message)
        print(f"Slack drift alert posted. Thread ID: {thread_id}")
        return f"Drift alert sent. Thread ID: {thread_id}"
    else:
        print("No drift detected - skipping Slack notification")
        return "No drift detected"


@pipeline
def price_prediction_evaluation(
    n_samples: int = 1000,
    model_artifact: str = "price_prediction_model",
    simulate_drift: bool = False
) -> Tuple[
    Annotated[Dict, "evaluation_metrics"],
    Annotated[HTMLString, "drift_report"]
]:
    """
    Evaluation pipeline that loads evaluation dataset with ground truth,
    evaluates the production model, creates a drift report, and attaches metadata to the model.
    
    Args:
        n_samples: Number of samples in evaluation dataset
        model_artifact: Name of the model artifact to evaluate
        simulate_drift: If True, simulate data drift in the evaluation dataset
    """
    # Load reference (baseline) data without drift
    reference_data = load_reference_data(n_samples)
    
    # Load evaluation dataset with ground truth (potentially with drift)
    evaluation_data = load_evaluation_dataset(n_samples, simulate_drift=simulate_drift)
    
    # Preprocess both datasets
    preprocessed_reference = preprocess_evaluation_data(reference_data)
    preprocessed_evaluation = preprocess_evaluation_data(evaluation_data)
    
    # Load production model
    production_model = load_production_model(model_artifact)
    
    # Generate predictions on both datasets
    reference_predictions_df = generate_predictions(preprocessed_reference, production_model)
    evaluation_predictions_df = generate_predictions(preprocessed_evaluation, production_model)
    
    # Calculate evaluation metrics on evaluation data
    evaluation_metrics = calculate_evaluation_metrics(evaluation_predictions_df)
    
    # Create model performance/drift report (compare predictions on reference vs evaluation)
    # This will detect model drift by comparing how the model performs on reference vs drifted data
    model_drift_report_json, model_drift_report_html = model_performance_step(
        reference_dataset=reference_predictions_df,
        comparison_dataset=evaluation_predictions_df
    )
    
    # Check for drift and send Slack alert if detected
    # drift_alert_result = slack_drift_alert(model_drift_report_json)
    
    # Attach metadata to the model
    metadata_result = attach_metadata_to_model(evaluation_metrics, evaluation_predictions_df)
    
    return evaluation_metrics, model_drift_report_html


@click.command()
@click.option("--config", default="evaluation_config.yaml", help="Path to configuration file")
def main(config: str):
    """Run the price prediction evaluation pipeline."""
    price_prediction_evaluation.with_options(config_path=config)()


if __name__ == "__main__":
    main()

