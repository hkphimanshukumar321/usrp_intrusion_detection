# Figures and Model Summaries: SDR Intrusion Detection

## 1. What are "Spike Detect" and "MLP Baseline"?

Before diving into the figures, it's important to define the baseline models we are comparing our custom CNNs against.

### Spike Detect (Traditional Baseline)
**What it is:** This is the non-Machine Learning traditional baseline. It is a direct port of the GNU Radio Flowgraph block (`gr::blocks::peak_detector_fb`). 
**How it works:** It processes the raw IQ signal, calculates its magnitude, and maintains a running average. When a signal peak exceeds a certain threshold (accounting for `alpha` running average, rise/fall times, and hysteresis), it triggers a "1" (a spike). The algorithm then calculates the "Spike Density" over a full window and uses nearest-neighbor statistical thresholds boundaries to guess the classification (Clear, Human, Animal, Drone).
**Why use it:** This operates as the absolute baseline. If Deep Learning models cannot beat this, then applying neural networks to the SDR is computationally wasteful. (Spoiler: the DL models completely outperform it).

### MLP Baseline (Simplest Deep Learning Baseline)
**What it is:** MLP stands for Multi-Layer Perceptron. It is standard "dense" or "feed-forward" neural network.
**How it works:** Instead of looking at time series steps or 2D spectrograms, it simply takes the 256x2 (I & Q) window and flattens it into a 512-element 1D array. It then passes this array through standard Dense Layers (`Linear → ReLU → Dropout`). 
**Why use it:** It shows what happens when you throw data at a neural network *without* inductive biases. A CNN inherently understands spatial and temporal relationships (what came exactly before and exactly after). An MLP treats every data point totally independently. The ~47% accuracy indicates the model struggles to parse the complex RF phase/time relationships without convolutions.

---

## 2. Explanation of the Generated Figures

All figures represent the data synthesized during our ablation study over the SDR datasets.

### `fig01_accuracy_comparison` (Horizontal Bar Chart)
* **What it is:** A comparison of the mean accuracy of all evaluated models, sorted from worst to best, featuring error bars.
* **Why it matters:** This is the primary "headline" graph. It immediately demonstrates that the `1D-CNN (IQ)` and `Dual-Branch Fusion` models crush the classical `Spike Detect`, traditional `SVM`, and `MLP` Baselines.

### `fig02_confusion_matrices` (Grid of Grid Matrices)
* **What it is:** Normalized matrices demonstrating what each model predicted versus the true class. Darker blue means more correct predictions on the diagonal.
* **How to read:** The true label is on the y-axis, predicted label on the x-axis. A 1.00 on the diagonal means 100% perfect classification for that class. It reveals where models get explicitly "confused" (e.g., mistaking a drone for animal interference).

### `fig03_training_curves` (Line Plots)
* **What it is:** Left graph shows Training Loss decreasing over epochs. Right graph shows Validation Accuracy increasing over epochs.
* **Why it matters:** Shows the learning efficiency. You can see which models converge quickly and which ones bounce around unstably. It also verifies that the models are not wildly overfitting to the training set.

### `fig04_f1_heatmap` (Class vs Model Heatmap)
* **What it is:** An overarching view of the F1-Score (balance of precision and recall) for each model against each specific class.
* **How to read:** Finding a dark red square means a particular model is fantastic at detecting that specific class. Even worse-performing models might be surprisingly adept at a singular class.

### `fig05_radar_chart` (Spider / Hex Plot)
* **What it is:** Shows multiple metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1) mapped simultaneously on a circular plot.
* **How to read:** The wider and more symmetrical the "web" stretches toward the outer ring (1.0), the better the overall, unbiased performance of the model.

### `fig06_pareto_accuracy_params` (Log-Scatter with Trendline)
* **What it is:** Mean Accuracy (y-axis) mapped against the Number of Training Parameters in log-scale (x-axis). 
* **Why it matters:** In IoT and Edge computing, model size directly impacts latency, battery, and feasibility. The dashed "Pareto Front" line connects the models creating the absolute best trade-offs between size and performance. Any model to the far right but low on the y-axis is very inefficient. Our `Dual-Branch Lite` is a great pareto-performer here.

### `fig07_kfold_boxplot` (Box and Whisker Plots)
* **What it is:** A view of the accuracy distribution of the 5 cross-validation folds.
* **How to read:** A very "tall" or split box means the model is unstable—it performed great on one dataset split but terribly on another. A tight, compressed box high on the y-axis indicates a highly reliable, deterministic model.

### `fig08_precision_recall_bars` (Paired Grouped Bar Chart)
* **What it is:** Grouped bars breaking down purely Precision (false positive resistance) and Recall (false negative resistance).
* **Why it matters:** In intrusion detection systems, false alarms cause fatigue, but missed alarms are a critical security failure. This dissects how each model handles that trade-off.

### `fig09_roc_curves` (ROC AUC Line Charts)
* **What it is:** Determines how well a model can separate the positive vs negative instances (True Positive Rate vs False Positive Rate) taking a One-vs-Rest approach per class.
* **How to read:** The closer the line stays to the top-left corner, the better. The diagonal dashed line represents random guessing (0.50). Our deep learning models form tight squares in the top left, yielding ~0.95+ AUC calculations.

### `fig10_best_model_fold_curves` (Shaded Mean Path)
* **What it is:** The training and validation curves isolated for the absolute *best* performing model across all 5 of its folds.
* **How it read:** It calculates the mean path through the center and paints a shaded ±1 Standard Deviation area around it to show the absolute convergence bounds.

### `fig11_summary_table` (Pre-formatted Data Table)
* **What it is:** A cleanly formatted IEEE-styled results table comparing Model, Parameters, Accuracy, and F1 (Macro).
* **Why it matters:** Ready to be directly integrated into LaTeX documents or presentation slide-decks without formatting hassle.

### `fig12_class_accuracy_breakdown` (Grouped Bar Chart)
* **What it is:** Groups performance by Model on the x-axis, separating the bars individually by Class color (Clear, Human, Animal, Drone).
* **How to read:** Immediately identifies if a model's high average accuracy is actually due to dominating 3 classes but entirely failing on a 4th. (E.g. Is it failing to catch drones?)

### `fig13_pareto_f1_params` (Bubble Scatter Chart)
* **What it is:** Similar to the Pareto optimization chart, but uses F1-Score on the y-axis, and turns the scatter dots into "bubbles" sized proportionally to the overall accuracy percentage of the model. Allows evaluating 3 variables dynamically.
