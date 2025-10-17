import torch
import torch.nn as nn
import joblib
import numpy as np
import json
import os
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
    def __init__(self, input_dim, hidden_layers, dropout_rate, activation="ReLU"):
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
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 4))  # 4 binary outputs
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoPilotNNMsgProcessor:
    def __init__(self, frame_interval=3, speed_threshold=10.0):
        self.frame_interval = (
            frame_interval  # how many frames between forced forward presses
        )
        self.speed_threshold = (
            speed_threshold  # speed must be below this to force forward
        )
        self.frame_count = 0

        model_dir = os.path.join(os.path.dirname(__file__), "../model")
        # Load scaler
        self.scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.joblib"))
        if hasattr(self.scaler_X, "feature_names_in_"):
            del self.scaler_X.feature_names_in_
        # Load model config
        with open(os.path.join(model_dir, "model_config.json"), "r") as f:
            config = json.load(f)
        # Load model
        self.model = DrivingPolicy(
            input_dim=config["input_dim"],
            hidden_layers=config["hidden_layers"],
            dropout_rate=config["dropout_rate"],
            activation=config["activation"],
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "driving_policy.pth"),
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()

    def nn_infer(self, message):
        # Prepare features: raycasts + car speed
        features = list(message.raycast_distances) + [message.car_speed]
        features_scaled = self.scaler_X.transform([features])
        x_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(0).numpy().astype(int)
        return preds  # preds: [forward, back, left, right]

    def process_message(self, message, data_collector):

        preds = self.nn_infer(message)
        car_speed = message.car_speed

        # Override forward command every frame_interval if speed below threshold
        forward_cmd = preds[0]
        if self.frame_count >= self.frame_interval:
            if car_speed < self.speed_threshold:
                forward_cmd = True
            elif car_speed >= self.speed_threshold:
                self.frame_count == 0
        else:
            self.frame_count += 1
        commands = [
            ("forward", forward_cmd),
            ("back", bool(preds[1])),
            ("left", bool(preds[2])),
            ("right", bool(preds[3])),
        ]

        for command, active in commands:
            data_collector.onCarControlled(command, active)


if __name__ == "__main__":
    import sys

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = AutoPilotNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
