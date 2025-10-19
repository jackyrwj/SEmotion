# SEmotion: Knowledge-guided Sigmoid-Constrained Network for EEG-based Emotion Recognition

This repository contains the implementation of the paper **"SEmotion: Knowledge-guided Sigmoid-Constrained Network for EEG-based Emotion Recognition"**. The project focuses on leveraging EEG signals for emotion recognition using a novel sigmoid-constrained network guided by domain knowledge.

## Abstract
EEG-based emotion recognition is a challenging task due to the variability in EEG signals across subjects and sessions. In this work, we propose SEmotion, a knowledge-guided sigmoid-constrained network that incorporates domain knowledge to improve the generalization and robustness of emotion recognition models. The method is evaluated on benchmark EEG datasets, demonstrating state-of-the-art performance in cross-subject and cross-session scenarios.

## Datasets
The datasets used in this study include:

- **SEED**: A widely-used dataset for EEG-based emotion recognition.
- **SEED-IV**: An extension of SEED with four emotion categories.

You can download the datasets from the [BCMI official website](https://bcmi.sjtu.edu.cn/~seed/index.html).

## Usage
To run the experiments, execute the following command:

```bash
python SEmotion/msmdaer.py
```

The results will be printed in the terminal.

## Results
The proposed SEmotion model achieves state-of-the-art performance on both SEED and SEED-IV datasets. Detailed results and comparisons with baseline methods are provided in the paper.

## Citation
If you find this work useful, please consider citing our paper:

```bibtex
@article{your2025semotion,
  title={SEmotion: Knowledge-guided Sigmoid-Constrained Network for EEG-based Emotion Recognition},
  author={Your Name and Collaborators},
  journal={Journal/Conference Name},
  year={2025}
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
We thank the authors of the SEED and SEED-IV datasets for making their data publicly available.

