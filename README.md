# Downstream-Experiments
# Downstream Experiments with MixUp and JTT

This repository explores advanced data augmentation techniques, such as **MixUp**, and methods like **Just Train Twice (JTT)** to improve fairness and robustness in machine learning models. The project evaluates performance on challenging datasets and investigates strategies to mitigate spurious correlations.

## Key Features
- Implementation of **MixUp** and **JTT** for robust training.
- Experiments on balanced and unbalanced datasets:
  - **CelebA**: Balanced and unbalanced class ratios.
  - **Waterbird**: Datasets with spurious correlations.
  - Additional datasets: **Color MNIST**, **Jigsaw**, etc.
- Performance evaluation based on:
  - Average accuracy.
  - Accuracy across majority/minority groups.

## Results
- Comparative analysis of methods, including **ERM**, **DRO**, **JTT**, and **GIC**.
- Insights into mitigating overfitting and enhancing generalization.
- Improved handling of imbalanced and biased data distributions.

## Citation
If you use this work, please cite the MixUp and JTT papers referenced in this repository.

---
