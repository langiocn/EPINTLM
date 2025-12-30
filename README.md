---

````markdown
This repository contains the implementation of "EPINTLM: Residual Cross-Attention for Enhancer–Promoter Prediction with Pretrained k-mer Embeddings" for predicting enhancer–promoter interactions.

## Downloads

| Resource | Link | Folder to place |
|---------|------|----------------|
| Model checkpoints | https://drive.google.com/drive/folders/18DHZgsJqupNTnWmPrRiA3F1SrMro2q_H?usp=sharing | ./checkpoints/ |

---

## Feature Extraction

Before running the command below, make sure to edit the input file paths (FASTA, BED, and label files) for the specific cell line you want to evaluate in `feature_extraction/seqgendataset.py`.

```bash
python ./feature_extraction/seqgendataset.py
````

---

## Run Test

```bash
python testepintlm.py
```

