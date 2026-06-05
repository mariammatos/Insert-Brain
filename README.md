# Insert Brain

Brain-computer interface system that acquires EEG signals, trains classifiers, and controls a robot in real-time.

## Overview

This project combines EEG signal processing with machine learning to create a real-time BCI (brain-computer interface). It includes:

- **EEG Acquisition**: Real-time streaming and recording of EEG data
- **Model Training**: Building subject-specific and multi-session classifiers
- **Real-time Control**: Using trained models to control a robot based on EEG signals
- **Data Visualization**: Exploring and analyzing EEG sessions and trained models

## Project Structure

```
project/
├── main_recording_eeg.py          # Main script for EEG recording
├── realtime_robot_control.py      # Real-time BCI-based robot control
├── offline_playback_robot_control.py  # Replay recorded EEG for testing
├── test_model.py                  # Test trained classifiers
├── config.py                      # Configuration settings
├── acquisition/                   # EEG data acquisition modules
├── protocol/                      # Experimental protocols and stimuli
├── training/                      # Model training scripts
└── visualize/                     # Data exploration and visualization tools

data/                              # Recorded sessions and trained models
insert_brain_connected/            # Arduino firmware for hardware control
```

## Quick Start

Each script includes inline documentation with detailed instructions. Key entry points:

1. **Record EEG Session**: `python project/main_recording_eeg.py`
   - Runs the Motor Imagery protocol and automatically trains a subject-specific classifier at the end
   
2. **Run Real-time BCI Control**: `python project/realtime_robot_control.py <session_path>`
   - Streams EEG in real-time, runs the cascade classifier, and sends commands to the robot
   - **Before running:** Upload `insert_brain_connected/insert_brain_connected.ino` to the Arduino
   
3. **Explore Data**: `python project/visualize/explore_session.py`

## Configuration

Update `project/config.py` to set:
- Sampling rate and hardware parameters (serial port, board ID)
- Model paths and hyperparameters
- Data directories

## Arduino Setup

Before running real-time robot control, upload the firmware to the Arduino:

1. Open `insert_brain_connected/insert_brain_connected.ino` in Arduino IDE
2. Select the correct board and serial port
3. Upload to the Arduino device

This is required before any session using `realtime_robot_control.py`, `manual_robot_control.py` or `offline_playback_robot_control.py`.

## PhysioNet Demo Dataset

The notebook `project/PhysioNet_Cascade_DemoSession.ipynb` processes the first 50 subjects from the PhysioNet EEG Motor Movement/Imagery Database to:

- Train and evaluate cascaded classifiers (GATING → AXIS → DIRECTION)
- Split each subject's data into 80% training / 20% holdout
- Generate demo sessions using the holdout data
- Create evaluation reports

**Note:** This notebook works in both Google Colab and VS Code:
- **Colab**: Saves results to Google Drive (`/content/drive/MyDrive/BCI_RoboticArm/HoldoutDemo`)
- **VS Code**: Saves locally to `./holdout_demo_sessions`

## Notes

- Most utility scripts are helper modules for the main workflows
- Detailed usage instructions are included at the beginning of each file
- The `protocol/` module defines experimental paradigms and stimulus presentation
- Trained models are stored in `data/` with associated metadata

## Hardware

- **EEG Headset**: OpenBCI Cyton Board (8-channel)
- **Amplifier**: Connected via serial port (configurable in `config.py`)
- **Robot Arm**: Controlled via Arduino with servo motors
  - Firmware: `insert_brain_connected/insert_brain_connected.ino`
  - Accepts both real time BCI protocol commands (`realtime_robot_control.py`), previously recorded EEG data (`offline_playback_robot_control.py`) and manual control signals (`manual_robot_control.py`)

---

*For detailed documentation on specific modules, see inline comments in each file.*
