# EPINTLM: Enhancer–Promoter Prediction with Pretrained k-mer Embeddings and Residual Cross-Attention

**This repository contains the implementation of EPINTLM for predicting enhancer–promoter interactions.**

## 1. Downloads
Download the pretrained EPINTLM checkpoint required for evaluation and copy it into the `./checkpoints/` folder:
| Resource | Link | Folder to place |
|---------|------|----------------|
| Model checkpoints | https://drive.google.com/drive/folders/18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H?usp=sharing | ./checkpoints/ |

---

## 2. Feature Extraction

Before running the command below, make sure to edit the input file paths (FASTA, BED, and label files) for the specific cell line you want to evaluate in `feature_extraction/seqgendataset.py`.

```bash
python ./feature_extraction/seqgendataset.py
````

---

## 3. Run Test
Before running the command below, make sure to edit the checkpoint path in `testepintlm.py` to point to the correct pretrained model you want to evaluate.
```bash
python testepintlm.py
```

