# ⚽ Football Analysis

<img src="images/demo_output.gif" width="600"/>

A computer vision project for tracking football players in videos using **YOLOv5** and **K-Means Clustering**. Trained on a custom dataset created via **RoboFlow**.



## 🔍 Overview

- **Player Detection:**  
  Custom-trained YOLOv5 model detects players in each frame.

- **Tracking & ID Assignment:**  
  Uses bounding box centroids and **K-Means clustering** to assign consistent player identities across frames.

- **Dataset Creation:**  
  Dataset annotated and exported using **RoboFlow**.



## 🛠️ Key Components

- YOLOv5 detection and tracking pipeline  
- K-Means clustering for player ID assignment  
- Visual output with bounding boxes and player labels



## 🎯 Applications

- Automated player tracking  
- Positional and movement analysis  
- Extendable to other sports or tracking tasks
