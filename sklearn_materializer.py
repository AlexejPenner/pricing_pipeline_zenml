import os
import joblib
from typing import Type, Any, Dict
from sklearn.pipeline import Pipeline
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType, VisualizationType
from zenml.metadata.metadata_types import MetadataType


class SklearnPipelineMaterializer(BaseMaterializer):
    """Custom materializer for sklearn Pipeline objects."""
    
    # Define the data types this materializer can handle
    ASSOCIATED_TYPES = (Pipeline,)
    
    # Define the artifact type
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    
    def load(self, data_type: Type[Any]) -> Pipeline:
        """Load sklearn Pipeline from storage."""
        filepath = os.path.join(self.uri, "model.joblib")
        with self.artifact_store.open(filepath, "rb") as f:
            # Use joblib to load the pipeline (better than pickle for sklearn)
            pipeline = joblib.load(f)
        return pipeline
    
    def save(self, pipeline: Pipeline) -> None:
        """Save sklearn Pipeline to storage."""
        filepath = os.path.join(self.uri, "model.joblib")
        with self.artifact_store.open(filepath, "wb") as f:
            # Use joblib to save the pipeline (better than pickle for sklearn)
            joblib.dump(pipeline, f, compress=3)  # compress=3 for good balance of speed/size
    
    def save_visualizations(self, pipeline: Pipeline) -> Dict[str, VisualizationType]:
        """Generate visualizations for the pipeline."""
        visualizations = {}
        
        # Create a simple HTML visualization of the pipeline structure
        vis_path = os.path.join(self.uri, "pipeline_structure.html")
        
        html_content = self._generate_pipeline_html(pipeline)
        
        with self.artifact_store.open(vis_path, "w") as f:
            f.write(html_content)
        
        visualizations[vis_path] = VisualizationType.HTML
        return visualizations
    
    def extract_metadata(self, pipeline: Pipeline) -> Dict[str, MetadataType]:
        """Extract metadata from the sklearn Pipeline."""
        metadata = {}
        
        # Extract basic pipeline information
        metadata["num_steps"] = len(pipeline.steps)
        metadata["step_names"] = [step[0] for step in pipeline.steps]
        metadata["pipeline_class"] = pipeline.__class__.__name__
        
        # Get the final estimator info
        if hasattr(pipeline, 'named_steps') and pipeline.steps:
            final_step_name, final_estimator = pipeline.steps[-1]
            metadata["final_estimator"] = final_estimator.__class__.__name__
            metadata["final_step_name"] = final_step_name
            
            # If the final estimator has feature_importances_, include that info
            if hasattr(final_estimator, 'feature_importances_'):
                metadata["has_feature_importances"] = True
                
            # If it's a fitted estimator, try to get more info
            if hasattr(final_estimator, 'n_features_in_'):
                metadata["n_features_in"] = int(final_estimator.n_features_in_)
                
        return metadata
    
    def _generate_pipeline_html(self, pipeline: Pipeline) -> str:
        """Generate HTML visualization of pipeline structure."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>sklearn Pipeline Structure</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .step { 
                    border: 2px solid #3498db; 
                    margin: 10px 0; 
                    padding: 15px; 
                    border-radius: 8px;
                    background-color: #f8f9fa;
                }
                .step-name { 
                    font-weight: bold; 
                    color: #2c3e50; 
                    font-size: 18px;
                    margin-bottom: 5px;
                }
                .step-class { 
                    color: #7f8c8d; 
                    font-style: italic;
                    margin-bottom: 10px;
                }
                .step-params {
                    font-size: 12px;
                    color: #34495e;
                    background-color: #ecf0f1;
                    padding: 5px;
                    border-radius: 4px;
                }
                .arrow {
                    text-align: center;
                    font-size: 24px;
                    color: #3498db;
                    margin: 5px 0;
                }
            </style>
        </head>
        <body>
            <h1>sklearn Pipeline Structure</h1>
        """
        
        for i, (step_name, estimator) in enumerate(pipeline.steps):
            html += f"""
            <div class="step">
                <div class="step-name">Step {i+1}: {step_name}</div>
                <div class="step-class">{estimator.__class__.__module__}.{estimator.__class__.__name__}</div>
                <div class="step-params">
                    Parameters: {str(estimator.get_params())[:200]}{'...' if len(str(estimator.get_params())) > 200 else ''}
                </div>
            </div>
            """
            if i < len(pipeline.steps) - 1:
                html += '<div class="arrow">↓</div>'
        
        html += """
        </body>
        </html>
        """
        return html