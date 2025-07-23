# ⚽ Football Analysis

**Football Analysis: Player Tracking & Assignment using YOLOv5 and K‑Means Clustering**  
A Python-based computer vision pipeline for football analytics — tracking players across frames and assigning player identities using clustering algorithms.



## 🔍 Overview

This project implements:

- **Player Detection & Tracking**  
  Detects players in video frames using a custom-trained **YOLOv5** model.

- **Player Assignment via K‑Means Clustering**  
  Associates each detected player across frames with unique identities by clustering their bounding-box centroids and appearance features using **K‑Means**.

- **Dataset Creation with RoboFlow**  
  Prepared and annotated the training dataset using **RoboFlow**, then exported it for fine-tuning YOLOv5.



## 🛠️ Key Components

- **YOLOv5-based Detection**  
  A custom dataset, exported from RoboFlow, is used to train the YOLOv5 detection model. The model outputs bounding boxes and confidence scores for each player in frame.

- **Tracking & ID Assignment**  
  - Extracts bounding-box centroids and optional appearance descriptors per detection.  
  - Applies **K‑Means clustering** over time windows to group detections and assign stable identities to players.

- **Utility Scripts**  
  - Data loaders and annotation processing scripts from RoboFlow exports  
  - Inference pipeline that detects, tracks, clusters, and visualizes player IDs overlaid on frames



## 🎯 Goals & Applications

- **Automated Player Tracking**  
  Valuable for sports analytics, automating player movement analysis and heatmaps.

- **Team Performance Metrics**  
  Enables statistical insights like distance covered, average positioning, and player interactions.

- **Extendable Framework**  
  Easy to adapt to other sports, player-specific modeling, or sophisticated tracking methods.
