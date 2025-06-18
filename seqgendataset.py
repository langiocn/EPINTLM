import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import itertools

class SeqGenDataset(Dataset):
    def __init__(self, enhancer_ids, promoter_ids, gene_data_all, labels):
        self.enhancer_ids = enhancer_ids
        self.promoter_ids = promoter_ids
        self.gene_data_all = gene_data_all
        self.labels = labels
        
    def __getitem__(self, idx):

        enhancer_id = self.enhancer_ids[idx]
        promoter_id = self.promoter_ids[idx]
        label = self.labels[idx]
        gene_data = self.gene_data_all[idx]
        
        return enhancer_id.squeeze(), promoter_id.squeeze(), gene_data, label
    
    def __len__(self):
        return len(self.labels)

def create_tokenizer(k=6):
    f = ['A','C','G','T']
    token_dict = {''.join(c): idx for idx, c in enumerate(itertools.product(f, repeat=k))}
    token_dict['null'] = 4096  
    
    def tokenizer(sequence):
        tokens = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        token_ids = [token_dict.get(token, 0) for token in tokens]
        return torch.tensor(token_ids)
    
    return tokenizer

def load_fasta(file_path, tokenizer, num_workers=8):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith(">")]
    
    print("Tokenizer...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tokenized = list(executor.map(tokenizer, lines))

    tokenized = np.array(tokenized, dtype=np.int64) 
    return torch.from_numpy(tokenized)


def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file]

    labels_array = np.array(labels)

    return labels_array

# Tạo tokenizer
tokenizer = create_tokenizer()

# Load dữ liệu enhancer và promoter từ fasta
print("Loading enhancer and promoter...")
enhancer_ids = load_fasta('./final_dataset/HeLa-S3/HeLa-S3_enhancer.fasta', tokenizer)
promoter_ids = load_fasta('./final_dataset/HeLa-S3/HeLa-S3_promoter.fasta', tokenizer)

# Load labels
print("Loading labels...")
labels = load_labels('/content/EPIPDLF/final_dataset/HeLa-S3/HeLa-S3_label.txt')

# Load gene_data từ GenomicFeatures
from genomic_features import GenomicFeatures
gene_data = GenomicFeatures(
    enh_datasets="./data/bed_files/HeLa-S3/HeLa-S3_enhancer.bed",
    pro_datasets="./data/bed_files/HeLa-S3/HeLa-S3_promoter.bed",
    feats_config="./data/configfiles/CTCF_DNase_6histone.500.json",
    feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
    cell="HeLa",
    enh_seq_len=3000,
    pro_seq_len=2500,
    bin_size=500
)


combined_dataset = SeqGenDataset(enhancer_ids, promoter_ids, gene_data, labels)
dataloader = DataLoader(dataset=combined_dataset, batch_size=128)
torch.save(combined_dataset, './data/nu_HeLa_combined_dataset.pt')
