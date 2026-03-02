# Price Prediction ML Pipeline

## Quick Start

1. Install dependencies:
```bash
pip install uv
uv pip install -r requirements.txt
```

2. Initialize you ZenML repo
```bash
zenml init
zenml login
zenml project set <INSERT_NAME>
zenml stack set <INSERT_NAME>
```

2. Run training pipeline:
```bash
# Hydra-based configuration (default: training pipeline)
python training_pipeline.py

# Select pipeline variant via Hydra
python training_pipeline.py pipeline=drift_no_drift
python training_pipeline.py pipeline=drift_with_drift
python training_pipeline.py pipeline=training_schedule

# Override parameters from the command line
python training_pipeline.py parameters.epochs=20 parameters.country=Germany

# Or use a raw YAML config file (bypasses Hydra)
python training_pipeline.py --config path/to/your/config.yaml
```

3. Run inference:
```bash
python inference_pipeline.py
```

### Deploy Inference with Web UI

Deploy the inference pipeline as a long-running HTTP service with a built-in visual playground (table row input + prediction):


via CLI:
```bash
zenml pipeline deploy inference_pipeline.price_prediction_inference \
  --name price_prediction \
  --config config_old/inference_deploy.yaml
```

Once deployed, open the deployment URL (e.g. `http://localhost:8000`) to use the web UI: edit the pre-filled row, click **Predict**, and see the predicted price on the same page.

- **Web UI**: `{deployment_url}/`
- **API docs**: `{deployment_url}/docs`
- **Invoke via CLI**: `zenml deployment invoke price_prediction --category=Electronics --brand_rating=4.5 ...`

## Schedule Management

### Deploy Scheduled Pipeline
```bash
# Create and deploy a scheduled training pipeline (runs every minute for testing)
python training_pipeline.py pipeline=training_schedule
# Or with legacy config file:
python training_pipeline.py --config config_old/training_schedule_config.yaml
```

### Manage Schedules
```bash
# List all schedules
zenml pipeline schedule list
# Update schedule cron expression
zenml pipeline schedule update price-prediction-training-schedule --cron-expression="*/2 * * * *"

# Delete a schedule (removes from both ZenML and orchestrator)
zenml pipeline schedule delete price-prediction-training-schedule
```


## Current Implementation

This is a proof-of-concept ML pipeline using ZenML for price prediction with synthetic e-commerce data.

## Configuration (Hydra)

Configs live in `config/`:
- `config/base/default.yaml` – shared defaults
- `config/pipeline/training.yaml` – standard training
- `config/pipeline/training_schedule.yaml` – scheduled runs
- `config/pipeline/drift_no_drift.yaml` – drift detection, no drift
- `config/pipeline/drift_with_drift.yaml` – drift detection with drift

## Potential Improvements

- Move pipeline steps into individual module files
- Move pipeline steps into individual module files
- Use Model stages to attach the inference pipeline only to the 
- Connect to real data sources instead of synthetic data
- Add proper configuration management and environment variables
- Implement comprehensive testing and validation
- Add proper logging and monitoring
- Separate business logic from infrastructure concerns
- Add input validation and error handling
- Implement proper CI/CD pipeline