import torch
import torch.nn as nn
import joblib
import numpy as np
import json
from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class DrivingPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="ReLU"):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "ReLU":
                layers.append(nn.ReLU())
            elif activation == "Tanh":
                layers.append(nn.Tanh())
            elif activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 6))  # 2 outputs × 3 classes each
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1, 2, 3)


class ExampleNNMsgProcessor:
    def __init__(self):
        self.always_forward = False
        # Load scaler
        self.scaler_X = joblib.load("./model/scaler_X.joblib")
        # Remove feature names to avoid warning when transforming numpy array
        if hasattr(self.scaler_X, "feature_names_in_"):
            del self.scaler_X.feature_names_in_
        # Load model config
        with open("./model/model_config.json", "r") as f:
            config = json.load(f)
        # Load model
        self.model = DrivingPolicy(
            input_dim=16,
            hidden_layers=config["hidden_layers"],
            activation=config["activation"],
        )
        self.model.load_state_dict(
            torch.load("./model/driving_policy.pth", map_location=torch.device("cpu"))
        )
        self.model.eval()

    def nn_infer(self, message):
        # Extract features: raycasts + car_speed
        features = list(message.raycast_distances) + [message.car_speed]
        features_scaled = self.scaler_X.transform([features])

        # To tensor
        x = torch.tensor(features_scaled, dtype=torch.float32)

        # Inference
        with torch.no_grad():
            logits = self.model(x)  # (1, 2, 3)
            preds = torch.argmax(logits, dim=2).squeeze(0).numpy()  # (2,)

        # Convert classes: 0 -> -1, 1 -> 0, 2 -> 1
        throttle_class = preds[0] - 1
        steer_class = preds[1] - 1

        commands = []
        # Throttle
        if throttle_class == 1:
            commands.append(("forward", True))
        elif throttle_class == -1:
            commands.append(("back", True))
        # Steer
        if steer_class == 1:
            commands.append(("right", True))
        elif steer_class == -1:
            commands.append(("left", True))

        print(f"Returning command {commands}")
        return commands

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    import sys

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
