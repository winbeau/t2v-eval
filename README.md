# T2V-Eval: Unified Evaluation for Text-to-Video Inference Strategies

This repository provides a **unified, no-reference evaluation pipeline** for **text-to-video (T2V) generation**, with a particular focus on **comparing different inference and acceleration strategies**.

The evaluation suite is designed for scenarios where **frame-aligned ground-truth videos are unavailable**, and thus relies on a set of complementary metrics that assess **semantic alignment, temporal quality, visual fidelity, and inference efficiency** directly from generated videos.

---

## Key Features

* **Semantic Alignment**

  * CLIPScore and VQAScore for measuring text–video consistency

* **Temporal Video Quality**

  * VBench (temporal-related dimensions) for high-level temporal coherence

* **Temporal Stability**

  * A no-reference **Temporal Flicker Score** to quantify frame-to-frame instability and boundary artifacts, particularly suitable for sliding-window and autoregressive inference

* **Visual Fidelity**

  * No-reference image quality metrics (NIQE / BRISQUE)

* **Efficiency & Scale**

  * Throughput (FPS) measurement under unified hardware and generation settings
  * Explicit reporting of generated frame count and video duration to ensure fair comparison

---

## Evaluation Protocol

All evaluation metrics are computed under a **unified preprocessing protocol**, ensuring fairness across different inference strategies:

* Fixed evaluation FPS
* Fixed number of frames per video
* Unified spatial resolution
* Uniform frame sampling
* Identical hardware and precision settings for runtime measurements

Generated videos are evaluated **without requiring real or reference videos**, making this repository suitable for **creative text-to-video generation** and **inference-time optimization research**.

---

## Supported Use Cases

* Comparison of **frame-level vs. head-level** inference strategies
* Analysis of **sliding-window length** and **attention head behaviors** (e.g., stable vs. oscillatory heads)
* Evaluation of **autoregressive diffusion transformer** acceleration methods
* Benchmarking inference efficiency–quality trade-offs in T2V models

---

## Metrics Overview

| Metric                 | Aspect                        | Reference Required |
| ---------------------- | ----------------------------- | ------------------ |
| CLIPScore / VQAScore ↑ | Text–video semantic alignment | No                 |
| VBench (Temporal) ↑    | High-level temporal quality   | No                 |
| Temporal Flicker ↓     | Frame-to-frame stability      | No                 |
| NIQE / BRISQUE ↓       | Visual fidelity               | No                 |
| FPS ↑                  | Inference efficiency          | No                 |
| #Frames / Duration     | Generation scale              | No                 |

---

## Design Philosophy

> **No ground truth required. No hidden assumptions. Reproducible by design.**

This repository emphasizes **clarity, reproducibility, and fairness**, making it suitable both for **academic research** and **practical benchmarking** of text-to-video inference strategies.


