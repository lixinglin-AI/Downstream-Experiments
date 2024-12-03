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
@article{zhang2018mixup,
    title={mixup: Beyond Empirical Risk Minimization},
    author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
    journal={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
@article{sagawa2020justtrain,
    title={Just Train Twice: Improving Group Robustness without Training Group Information},
    author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
    journal={International Conference on Machine Learning (ICML)},
    year={2020},
    url={https://github.com/anniesch/jtt},
}


---
