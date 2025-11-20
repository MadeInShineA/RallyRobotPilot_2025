from PyQt6 import QtWidgets
from data_collector import DataCollectionUI
import mlflow.pytorch
import torch
import numpy as np
from PIL import Image  # Using Pillow as per your preference/memory
import argparse

"""
This file implements an autopilot using a model trained and saved via MLflow.
It loads the model and uses it for inference within the DataCollectionUI framework.
/!\ Ensure the MLflow model path/URI is correct.
/!\ Be aware of processing time constraints as noted in the original example.
/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class MLflowModelNNMsgProcessor:
    def __init__(self, model_uri):
        """
        Initializes the processor by loading the MLflow model.

        Args:
            model_uri (str): The URI or path where the MLflow model is logged/stored.
                             e.g., "runs:/<run_id>/model", "./path/to/model", etc.
        """
        print(f"Loading model from MLflow URI: {model_uri}")
        try:
            self.model = mlflow.pytorch.load_model(
                model_uri, map_location="cpu"
            )  # Load on CPU for inference
            self.model.eval()  # Set model to evaluation mode
            self.input_channels = self.model.input_channels
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {model_uri}: {e}")
            raise e  # Re-raise to stop execution if model loading fails

        # Define the expected image dimensions based on your training pipeline
        # These should match the 'images_dimensions' used in your notebook (e.g., (90, 160))
        # It's good practice to store these alongside the model or pass them as arguments
        self.target_size = (
            self.model.input_height,
            self.model.input_width,
        )  # Use model's input dimensions
        self.class_names = [
            "forward",
            "back",
            "left",
            "right",
        ]  # Consistent with your notebook

        # Buffer to store previous preprocessed images
        self.image_buffer = []

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocesses a PIL image to match the format expected by the trained model.
        This replicates the preprocessing done in the notebook.
        """
        # Convert to grayscale if needed
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")

        # Resize using LANCZOS resampling (as in your notebook)
        pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_np = np.array(pil_image)

        # Normalize pixel values to [0, 1] range (as done in notebook's dataset)
        image_np = image_np.astype(np.float32) / 255.0

        # Add channel dimension (H, W) -> (1, H, W) for PyTorch
        image_np = np.expand_dims(image_np, axis=0)  # Shape becomes (1, H, W)

        # Convert to tensor (1, H, W)
        image_tensor = torch.from_numpy(image_np)

        return image_tensor

    def nn_infer(self, message):
        """
        Performs inference using the loaded MLflow model.

        Args:
            message: The input message, expected to contain an image accessible as message.image or similar.
                      The exact attribute name depends on the structure of your sensing_message.
                      Assumes message.image is a PIL Image or similar.
                      Adjust 'message.image' to the correct attribute name.

        Returns:
            list: A list of tuples representing commands, e.g., [("forward", True), ("left", False)].
                   Commands are active (True) if the model's probability exceeds a threshold (e.g., 0.5).
        """
        # --- Extract Image from Message ---
        # Adjust this line based on the actual structure of 'message'
        # Assuming message has an attribute like 'image', 'frame', or 'screenshot'
        pil_image = (
            Image.fromarray(message.image) if message.image is not None else None
        )  # Convert numpy array to PIL Image

        if pil_image is None:
            print("Warning: Received message with no image. Sending no commands.")
            return []

        # --- Preprocess the Image ---
        try:
            processed_image_tensor = self._preprocess_image(pil_image)  # (1, H, W)
        except Exception as e:
            print(f"Error preprocessing image: {e}. Sending no commands.")
            return []

        # --- Add to buffer ---
        self.image_buffer.append(processed_image_tensor)
        if len(self.image_buffer) > self.input_channels:
            self.image_buffer.pop(0)

        # If not enough frames, return no commands
        if len(self.image_buffer) < self.input_channels:
            print(
                f"Warning: Not enough frames in buffer ({len(self.image_buffer)}/{self.input_channels}). Sending no commands."
            )
            return []

        # --- Stack images along channel dimension ---
        stacked_tensor = torch.cat(self.image_buffer, dim=0)  # (input_channels, H, W)
        stacked_tensor = stacked_tensor.unsqueeze(0)  # (1, input_channels, H, W)

        # --- Run Model Inference ---
        with torch.no_grad():  # Disable gradient computation for efficiency
            try:
                logits_or_probs = self.model(stacked_tensor)
                # The model outputs raw logits. Apply sigmoid to get probabilities.
                probabilities = torch.sigmoid(logits_or_probs)
                # Detach from computation graph and move to CPU for numpy conversion
                prob_values = (
                    probabilities.detach().cpu().numpy()[0]
                )  # Remove batch dimension
            except Exception as e:
                print(f"Error during model inference: {e}. Sending no commands.")
                return []

        # --- Convert Probabilities to Commands ---
        commands = []
        threshold = (
            0.5  # Standard threshold for binary classification from logits/probs
        )
        for i, prob in enumerate(prob_values):
            action_name = self.class_names[i]
            active = bool(prob > threshold)
            commands.append((action_name, active))

        return commands

    def process_message(self, message, data_collector):
        """
        Processes the incoming sensing_message and sends commands based on model inference.
        """
        commands = self.nn_infer(message)
        print(commands)

        for command, active in commands:
            data_collector.onCarControlled(command, active)


if __name__ == "__main__":
    import sys
    import os

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    parser = argparse.ArgumentParser(description="Visual Autopilot with MLflow model")
    parser.add_argument(
        "model_path",  # Positional argument
        help="Path to the MLflow model directory (e.g., ./mlruns/891533531804789206/models/m-c3cab6f7f7da4101a6ba02e2ef9796f0)",
    )
    args = parser.parse_args()

    # Construct the full model URI by appending 'artifacts'
    model_uri = os.path.join(args.model_path, "artifacts")

    try:
        nn_brain = MLflowModelNNMsgProcessor(model_uri=model_uri)
    except Exception as e:
        print(f"Failed to initialize NN processor: {e}")
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
