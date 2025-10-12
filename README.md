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
# Using default configuration
python training_pipeline.py

# Using scheduled configuration (runs every minute for testing)
python training_pipeline.py --config training_schedule_config.yaml

# Using custom configuration
python training_pipeline.py --config path/to/your/config.yaml
```

3. Run inference:
```bash
python inference_pipeline.py
```

## Schedule Management

### Deploy Scheduled Pipeline
```bash
# Create and deploy a scheduled training pipeline (runs every minute for testing)
python training_pipeline.py --config training_schedule_config.yaml
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

## Potential Improvements

- Extract pipeline configurations from decorators to YAML files
- Move pipeline steps into individual module files
- Use Model stages to attach the inference pipeline only to the 
- Connect to real data sources instead of synthetic data
- Add proper configuration management and environment variables
- Implement comprehensive testing and validation
- Add proper logging and monitoring
- Separate business logic from infrastructure concerns
- Add input validation and error handling
- Implement proper CI/CD pipeline