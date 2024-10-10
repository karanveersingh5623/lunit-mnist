import os
import argparse
import mlflow
from fastapi import FastAPI, BackgroundTasks
from fastapi import File, UploadFile
from PIL import Image
import io
from urllib.parse import urlparse
import numpy as np

from src.data import load_mnist_data
from src.models import ImageClassifier
from src.trainer import Trainer
from src.utils import load_config, set_device
from src.checkpoints import save_checkpoint, load_checkpoint
from backend.models import DeleteApiData, TrainApiData, PredictApiData
import yaml
from typing import Optional
import mlflow.pyfunc
import logging

logger = logging.getLogger("mlflow")

# Set log level to debugging
logger.setLevel(logging.DEBUG)

# Load config.yaml
def load_config(file_path: str) -> dict:
    """Loads hyperparameters and configurations from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

app = FastAPI()
# Set tracking URI for mlflow
mlflow.set_tracking_uri("http://localhost:5001")

# mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = mlflow.tracking.MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())

current_registry_uri = mlflow.get_artifact_uri()
print("Current MLflow Registry URI: ", current_registry_uri)

# # Modify the train_model_task to load config from YAML
# def train_model_task(model_name: str, resume: bool):
#     """Train the model in the background with config loaded from YAML."""
#     config = load_config()  # Load config from the config.yaml file
    
#     epochs = config['epochs']
#     learning_rate = config['learning_rate']
#     batch_size = config['batch_size']
    
#     device = set_device()
#     mlflow.set_experiment("MNIST")

#     with mlflow.start_run() as run:
#         # Log parameters from the config.yaml
#         mlflow.log_params(config)

#         # Load data
#         train_dataloader, test_dataloader = load_mnist_data(batch_size)

#         # Initialize model and trainer
#         model = ImageClassifier().to(device)
#         trainer = Trainer(model, device, learning_rate)

#         # Load checkpoint if resuming
#         start_epoch = load_checkpoint(model, trainer.optimizer, config['checkpoint_path']) if resume else 0

#         for epoch in range(start_epoch, epochs):
#             trainer.train(train_dataloader, epoch)
#             accuracy, loss = trainer.evaluate(test_dataloader, epoch)

#             # Log metrics
#             mlflow.log_metric("accuracy", accuracy)
#             mlflow.log_metric("loss", loss)

#             # Save model checkpoints periodically
#             if (epoch + 1) % 5 == 0:
#                 save_checkpoint(model, trainer.optimizer, epoch, config['checkpoint_path'], run.info.run_id)

#         # Register the trained model
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         if tracking_url_type_store != "file":
#             mlflow.pytorch.log_model(
#                 model, "ImageClassifier", registered_model_name=model_name, conda_env=mlflow.pytorch.get_default_conda_env())
#         else:
#             mlflow.pytorch.log_model(
#                 model, "ImageClassifier-MNIST", registered_model_name=model_name)

#         # Transition to production stage
#         mv = mlflowclient.search_model_versions(
#             f"name='{model_name}'")[-1]  # Get the last model version
#         mlflowclient.transition_model_version_stage(
#             name=mv.name, version=mv.version, stage="production")

def train_model_task(model_name: str, resume: bool, config_file: Optional[str]):
    """Train the model in the background with config loaded from YAML."""
    # Load config from custom config file or default config.yaml
    config_path = config_file if config_file else "src/config.yaml"
    config = load_config(config_path)

    # epochs = config.get('epochs', 10)  # Default to 10 if not specified
    # learning_rate = config.get('learning_rate', 0.001)  # Default to 0.001 if not specified
    # batch_size = config.get('batch_size', 32)  # Default to 32 if not specified

    epochs = config['epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']

    device = set_device()
    mlflow.set_experiment("MNIST")

    with mlflow.start_run() as run:
        # Log parameters from the config.yaml
        mlflow.log_params(config)

        # Load data
        train_dataloader, test_dataloader = load_mnist_data(batch_size)

        # Initialize model and trainer
        model = ImageClassifier().to(device)
        trainer = Trainer(model, device, learning_rate)

        # Load checkpoint if resuming
        start_epoch = load_checkpoint(model, trainer.optimizer, config.get('checkpoint_path')) if resume else 0

        for epoch in range(start_epoch, epochs):
            trainer.train(train_dataloader, epoch)
            accuracy, loss = trainer.evaluate(test_dataloader, epoch)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("loss", loss)

            # Save model checkpoints periodically
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, trainer.optimizer, epoch, config.get('checkpoint_path'), run.info.run_id)

        # Register the trained model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(
                model, "ImageClassifier", registered_model_name=model_name, conda_env=mlflow.pytorch.get_default_conda_env())
        else:
            mlflow.pytorch.log_model(
                model, "ImageClassifier-MNIST", registered_model_name=model_name)

        # Transition to production stage
        mv = mlflowclient.search_model_versions(
            f"name='{model_name}'")[-1]  # Get the last model version
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production")

@app.get("/")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(), "Registry URI": mlflow.get_registry_uri()}

@app.post("/train")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Start the training process."""
    model_name = data.model_name
    resume = data.epochs > 0  # Logic for resume based on the epoch or other criteria
    config_file = data.config_file  # Custom config file path if provided

    # Add background task for training the model
    background_tasks.add_task(train_model_task, model_name, resume, config_file)
    return {"result": f"Training task for {model_name} started"}

@app.post("/predict")
async def predict_api(model_name: str, file: UploadFile = File(...)):
    """
    Predict on the provided image file.
    
    Parameters:
    - model_name: The name of the model to use for prediction.
    - file: The image file uploaded for prediction.
    """
    # Load the image from the uploaded file
    img = Image.open(io.BytesIO(await file.read())).convert('L')  # Convert image to grayscale if needed

    # Resize the image to the expected input size (28x28 for MNIST, for example)
    img = img.resize((28, 28))
    
    # Convert the image to a NumPy array and preprocess (flatten, normalize)
    img_np = np.array(img, dtype=np.float32).flatten()[np.newaxis, ...] / 255

    # Load the model from the MLflow registry
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/1")

    # Make a prediction
    pred = model.predict(img_np)
    
    # Postprocess the result
    predicted_class = int(np.argmax(pred[0]))  # Get the predicted class with the highest probability

    return {"predicted_class": predicted_class}


@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    """Delete model or model versions."""
    model_name = data.model_name
    version = data.model_version
    
    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {"result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {"result": f"Deleted version {version} of model {model_name}"}
    return response
