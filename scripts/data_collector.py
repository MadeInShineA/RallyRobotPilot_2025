import os.path

from rallyrobopilot import *

from PyQt6.QtCore import Qt, QTimer
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6 import uic

import pickle
import lzma
from typing import List


def segment_by_checkpoints(snapshots: List) -> List[List]:
    """
    Segment the snapshots into lists based on checkpoint passages.
    Returns a list of segments, where each segment is the data between checkpoints.
    """
    if not snapshots:
        return []

    segments = []
    current_segment = []
    last_checkpoint_count = 0

    for snapshot in snapshots:
        current_checkpoints = getattr(snapshot, "checkpoints_passed", 0)

        # If checkpoint count increased, start a new segment
        if current_checkpoints > last_checkpoint_count:
            if current_segment:
                segments.append(current_segment)
            current_segment = [snapshot]
            last_checkpoint_count = current_checkpoints
        else:
            current_segment.append(snapshot)

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    return segments


class DataCollectionUI(QtWidgets.QMainWindow):
    def __init__(self, message_processing_callback=None):
        super().__init__()

        uic.loadUi("scripts/DataCollector.ui", self)

        buttons = [
            self.forwardButton,
            self.backwardButton,
            self.rightButton,
            self.leftButton,
        ]
        self.command_directions = {
            "w": "forward",
            "s": "back",
            "d": "right",
            "a": "left",
        }

        self.forwardButton.pressed.connect(
            lambda: self.onCarControlled("forward", True)
        )
        self.forwardButton.released.connect(
            lambda: self.onCarControlled("forward", False)
        )

        self.backwardButton.pressed.connect(lambda: self.onCarControlled("back", True))
        self.backwardButton.released.connect(
            lambda: self.onCarControlled("back", False)
        )

        self.rightButton.pressed.connect(lambda: self.onCarControlled("right", True))
        self.rightButton.released.connect(lambda: self.onCarControlled("right", False))

        self.leftButton.pressed.connect(lambda: self.onCarControlled("left", True))
        self.leftButton.released.connect(lambda: self.onCarControlled("left", False))

        self.recordDataButton.clicked.connect(self.toggleRecord)
        self.resetButton.clicked.connect(self.resetNForget)

        self.autopiloting = False

        def toggle_autopilot():
            self.autopiloting = not self.autopiloting
            self.AutopilotButton.setText(
                "AutoPilot:\n" + ("ON" if self.autopiloting else "OFF")
            )

        self.AutopilotButton.clicked.connect(toggle_autopilot)
        self.message_processing_callback = message_processing_callback

        self.saveRecordButton.clicked.connect(self.saveRecord)

        self.network_interface = NetworkDataCmdInterface(self.collectMsg)

        self.timer = QTimer()
        self.timer.timeout.connect(self.network_interface.recv_msg)
        self.timer.start(25)

        self.saving_worker = None

        self.recording = False

        self.recorded_data = []

    def collectMsg(self, msg):
        if self.recording:
            if not self.saveImgCheckBox.isChecked():
                msg.image = None

            self.recorded_data.append(msg)
            self.nbrSnapshotSaved.setText(str(len(self.recorded_data)))

        if self.autopiloting:
            if self.message_processing_callback is not None:
                self.message_processing_callback(msg, self)

    def resetNForget(self):
        if len(self.recorded_data) == 0:
            return

        nbr_snapshots_to_forget = (
            self.forgetSnapshotNumber.value()
            if len(self.recorded_data) > self.forgetSnapshotNumber.value()
            else len(self.recorded_data) - 1
        )

        self.recorded_data = self.recorded_data[:-nbr_snapshots_to_forget]
        self.nbrSnapshotSaved.setText(str(len(self.recorded_data)))

        self.network_interface.send_cmd(
            "set position "
            + str(self.recorded_data[-1].car_position)[1:-1].replace(" ", "")
            + ";"
        )
        self.network_interface.send_cmd(
            "set rotation " + str(self.recorded_data[-1].car_angle) + ";"
        )
        self.network_interface.send_cmd("reset;")

        self.toggleRecord()

    def toggleRecord(self):
        self.recording = not self.recording
        self.recordDataButton.setText("Recording..." if self.recording else "Record")

        # Notify the game server about recording status
        # Server will only capture images when recording is active
        recording_status = "true" if self.recording else "false"
        self.network_interface.send_cmd(f"set recording {recording_status};")
        print(
            f"[DataCollector] Recording: {self.recording} - Images will {'BE' if self.recording else 'NOT be'} captured"
        )

    def onCarControlled(self, direction, start):
        command_types = ["release", "push"]
        self.network_interface.send_cmd(command_types[start] + " " + direction + ";")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            if key_text in self.command_directions:
                self.onCarControlled(self.command_directions[key_text], True)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if isinstance(event, QtGui.QKeyEvent):
            key_text = event.text()
            if key_text in self.command_directions:
                self.onCarControlled(self.command_directions[key_text], False)

    def saveRecord(self):
        if self.saving_worker is not None:
            print("[X] Already saving !")
            return

        if len(self.recorded_data) == 0:
            print("[X] No data to save !")
            return

        if self.recording:
            self.toggleRecord()

        self.saveRecordButton.setText("Saving ...")

        os.makedirs("./records", exist_ok=True)

        # Find next available record number
        record_name = "./records/record_%d.npz"
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1

        record_dir = f"./records/record_{fid}"
        os.makedirs(record_dir, exist_ok=True)
        os.makedirs(f"{record_dir}/segments", exist_ok=True)

        class ThreadedSaver(QtCore.QThread):
            def __init__(self, record_path, segment_dir, record_name, data):
                super().__init__()
                self.record_path = record_path
                self.segment_dir = segment_dir
                self.record_name = record_name
                self.data = data

            def run(self):
                # Save complete record
                with lzma.open(self.record_path, "wb") as f:
                    pickle.dump(self.data, f)

                # Segment and save segments
                segments = segment_by_checkpoints(self.data)
                if segments:
                    for i, segment in enumerate(segments):
                        segment_path = os.path.join(
                            self.segment_dir, f"{self.record_name}_segment_{i}.npz"
                        )
                        with lzma.open(segment_path, "wb") as f:
                            pickle.dump(segment, f)
                    print(f"Saved {len(segments)} segments to {self.segment_dir}")
                else:
                    print("No checkpoint passages detected - no segments created")

        self.saving_worker = ThreadedSaver(
            f"{record_dir}/record_{fid}.npz",
            f"{record_dir}/segments",
            f"record_{fid}",
            self.recorded_data,
        )
        self.recorded_data = []
        self.nbrSnapshotSaved.setText("0")
        self.saving_worker.finished.connect(self.onRecordSaveDone)
        self.saving_worker.start()

    def onRecordSaveDone(self):
        print("[+] Recorded data saved to", self.saving_worker.record_path)
        self.saving_worker = None
        self.saveRecordButton.setText("Save")


if __name__ == "__main__":
    import sys

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    data_window = DataCollectionUI()
    data_window.show()

    app.exec()
