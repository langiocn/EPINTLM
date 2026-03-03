# EPINTLM: Enhancer–Promoter Prediction with Pretrained k-mer Embeddings and Residual Cross-Attention

**This repository contains the implementation of EPINTLM for predicting enhancer–promoter interactions.**

## 1. Downloads
Download the pretrained EPINTLM checkpoint required for evaluation and copy it into the `./checkpoints/` folder:
| Resource | Link | Folder to place |
|---------|------|----------------|
| Model checkpoints | https://drive.google.com/drive/folders/18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H?usp=sharing | ./checkpoints/ |

---

## 2. Feature Extraction

Before running the command below, make sure to edit the input file paths (FASTA, BED, and label files) for the specific cell line you want to evaluate in `feature_extraction/seqgendataset.py`. This script will generate processed `.pt` files and save them into the `./data/` directory for use in model testing.

```bash
python ./feature_extraction/seqgendataset.py
````

---

## 3. Run Test
Before running the command below, make sure to edit the checkpoint path and the data .pt file paths in testepintlm.py so they correctly point to the pretrained model and feature files you want to evaluate.
```bash
python testepintlm.py
```

---

## Citation

Please cite our paper as follows:
```bibtex
Nguyen, T. L., Kha, H. Q., Nguyen, P. K., Le, M. H. N., Le, D. T., & Quoc Khanh Le, N. (2026).
EPINTLM: enhancer–promoter prediction with pretrained k-mer embeddings and residual cross-attention.
Briefings in Bioinformatics, 27(1), bbag064.

