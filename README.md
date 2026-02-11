# R6 Sword Combat Animation Generator (Machine Learning)
![Procedural Sword Animation Demo](demo.gif)

A machine learning–based system that generates procedural sword combat animations for Roblox R6 characters.

Given a directional slash pattern (e.g., diagonal 1→9, overhead 2→8), the model predicts realistic multi-joint keyframe sequences that can be directly imported into a game. The project explores applying neural networks to animation synthesis and real-time game workflows.

Built with PyTorch and trained on extracted Moon Animator keyframe data, with dataset augmentation to improve motion variety and generalization.

## How It Works
- Extracts and preprocesses R6 sword animation keyframes
- Augments data via mirroring and rotation to expand training samples
- Trains a neural network to predict joint movement from directional inputs
- Outputs animation data as JSON for easy game engine integration

## Features
- Procedural animation generation using machine learning
- Real-time animation preview via desktop GUI
- JSON export compatible with Roblox workflows
- Modular pipeline from data processing to inference

## Tech Stack
- Python
- PyTorch
- Pandas / NumPy
- Tkinter
- PyInstaller

## Motivation
This project was built to explore how machine learning can be used to generate game-ready animations, reduce manual keyframing, and support more dynamic combat systems in games.
