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
python training_pipeline.py
```

3. Run inference:
```bash
python inference_pipeline.py
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