# Classical Vision as a Data Optimizer for LLM-Based Object Recognition

### A Multi-Strategy Ensemble Pipeline for Retail Shelf Analysis

This repository contains the official implementation of the hybrid pipeline presented in the paper, designed to optimize object recognition in retail environments by using classical computer vision as a pre-processing stage for Large Language Models (LLMs).

---

## Overview

The system proposes a **"data optimization"** approach rather than traditional model training. It utilizes an ensemble of 9 classical segmentation algorithms executed in parallel to isolate products on shelves. This significantly reduces visual noise, enabling models like **GPT-4o** to identify SKUs with high precision in a zero-shot manner, eliminating the need for expensive fine-tuning or custom training.

## Authors
* **Sergio Alejandro Covarrubias Cázares**
* **Icker Villalón Lozoya**
* **Javier Yahir Juárez Arroyo**
* **Alejandro Vallejo Elizondo**
* **Nezih-Nieto Gutiérrez**

*School of Engineering and Sciences, Tecnológico de Monterrey, Mexico.*

---

## Key Features

* **Multi-Strategy Ensemble:** Parallel implementation of 9 distinct segmentation methods:
    * Watershed, MSER, K-Means, Convex Hull, Alpha Shapes, Split-and-Merge, Distance Transform, Region Growing, and Contour Approximation.
* **CPU-Only Optimization:** Designed for high-performance execution on **edge devices** without requiring GPU acceleration. Built entirely using OpenCV and NumPy.
* **7-Stage Post-processing Chain:**
    1.  Full-image bounding box removal.
    2.  Percentile-based noise filtering.
    3.  IoU (Intersection over Union) merging.
    4.  Aspect-ratio outlier removal.
    5.  Duplicate suppression.
    6.  Diagonal-size grouping.
    7.  Final coordinate refinement.
* **Zero-Shot Integration:** A ready-to-use pipeline that prepares optimized crops for LLM Vision APIs.

---

## Requirements

* **Python:** 3.9+
* **Libraries:** OpenCV (`opencv-python`), NumPy

### Installation
```bash
pip install opencv-python numpy
