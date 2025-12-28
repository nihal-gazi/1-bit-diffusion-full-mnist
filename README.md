
# Structural Dominance and Topological Stability in 1-Bit Diffusion Models

**An empirical investigation into the geometric interpolation regimes of binary-weight diffusion priors.**

![Project Banner](results/comparison_grid.png)
*(Top: 16-Bit Control | Bottom: 1-Bit Experimental. Note the structural preservation despite binary weights.)*

## 📄 Abstract
This work investigates the limits of extreme quantization in generative diffusion models by analyzing the representational behavior of a Residual U-Net constrained to 1-bit weights ($\{-1, 1\}$). While prior quantization research has largely focused on large language models, we show that binary-weight diffusion priors can successfully approximate complex multi-class image manifolds without catastrophic mode collapse.

Through controlled experiments on MNIST subsets (single class, binary class, and full ten-class), we observe a consistent trade-off between textural fidelity and structural robustness. Quantitative evaluation demonstrates that 1-bit quantization functions as a strong regularizer: the binarized model achieves a higher **Memorization Score** (Euclidean distance to nearest training neighbor) of **5.89** compared to the FP16 baseline’s **5.54**, indicating reduced overfitting.

Latent interpolation analysis further reveals a qualitative shift in generation dynamics, where 1-bit models exhibit continuous topological deformation ("curling") rather than the intensity-based interpolation ("fading") characteristic of high-precision networks. Although Frechet MNIST Distance increases under quantization (approximately **425** vs. **241** for FP16), reflecting degraded high-frequency detail, a classifier-based utility evaluation shows that semantic fidelity remains largely preserved, with the 1-bit model achieving a **Legibility Score** of **88.73%** versus **92.22%** for the baseline.

These results indicate that extreme quantization preferentially preserves global structure over local texture, suggesting that 1-bit diffusion models constitute a viable and highly compressed structural prior for generative modeling in resource-constrained environments.

## 📂 Repository Structure

```text
├── models/                     # Trained Model Weights
│   ├── gen_1bit.pth            # The 1-Bit ResUNet
│   ├── gen_16bit.pth           # The 16-Bit Control ResUNet
│   ├── judge_mnist.pth         # The CNN Classifier used for evaluation
│   └── model_tester.py         # Utility script to load and test models
│
├── results/                    # Visual Evidence
│   ├── comparison_grid.png     # Side-by-side comparison
│   ├── 16 bit.png              # 16-Bit sample grid
│   └── 1 bit.png               # 1-Bit sample grid
│
├── rigorous_bench_trainer.py   # Main Experiment: Computes FMD & Entropy
├── simple_bench_trainer.py     # Initial Experiment: Computes Memorization & Diversity
├── legibility_evaluation.py    # Utility Experiment: Computes Classifier Confidence
└── README.md

```

## 📊 Key Findings

### 1. Generalization vs. Memorization

We compared a standard **16-Bit (Float16) ResUNet** against a custom **1-Bit (Binary) ResUNet** trained on the full MNIST dataset (0-9).

| Metric | 16-Bit Control | 1-Bit Experimental | Interpretation |
| --- | --- | --- | --- |
| **Memorization Score**<br><br>*(Avg Dist to Nearest Neighbor)* | `5.54` | **`5.89`** | Higher is better. The 1-bit model copies *less* from the training set, acting as a regularizer. |
| **Diversity Score**<br><br>*(Avg Intra-Sample Dist)* | `9.84` | **`9.67`** | Comparable scores indicate the 1-bit model avoids mode collapse despite limited capacity. |

### 2. The "Explosion" vs. "Curling" Phenomenon

High-resolution latent walks reveal a fundamental difference in how the models interpolate concepts:

* **16-Bit Behavior ("Explosion"):** Relying on **intensity interpolation**. Segments of digits dissolve or cross-fade into new shapes.
* **1-Bit Behavior ("Curling"):** Relying on **geometric interpolation**. The model physically bends and curls the edges of the digit to reshape it, treating the object as a topologically cohesive solid.

### 3. Limitations & Trade-offs (The "Texture Tax")

To rigorously test the limits, we evaluated 2,000 generated samples using a standard CNN-based Frechet MNIST Distance (FMD).

| Metric | 16-Bit Control | 1-Bit Experimental | Gap |
| --- | --- | --- | --- |
| **FMD (Realism)** | `241.4` | `425.8` | **~1.7x Penalty** |
| **Legibility (Utility)** | `92.22%` | `88.73%` | **~3.5% Drop** |

**Conclusion:** The high FMD score confirms that 1-bit models cannot model high-frequency texture (gradients/noise), resulting in a "stencil-like" aesthetic. However, the negligible drop in Legibility proves that **semantic information remains intact**. The model sacrifices beauty, but keeps the truth.

## 🚀 Usage

### Installation

```bash
pip install torch torchvision scipy matplotlib tqdm numpy

```

### Reproducing Experiments

**1. Run the Rigorous FMD Benchmark**
Trains both models from scratch and computes Frechet Distance and Class Entropy.

```bash
python rigorous_bench_trainer.py

```

**2. Run the Utility (Legibility) Test**
Loads pre-trained models from the `models/` directory and evaluates classifier confidence.
*(Note: Ensure paths in the script point to your `models/` folder)*

```bash
python legibility_evaluation.py

```

## 📜 License

MIT License. Free for academic and commercial use with attribution.

