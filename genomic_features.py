import os
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from misc_utils import hg19_chromsize

class GenomicFeatures(Dataset):
    
    def __init__(self, enh_datasets, pro_datasets, feats_config: Dict[str, str], feats_order: List[str], cell, enh_seq_len: int=3000, pro_seq_len: int=2500, bin_size: int=500):
        self.enh_datasets = enh_datasets
        self.pro_datasets = pro_datasets
        self.cell = cell

        self.enh_seq_len = int(enh_seq_len)
        self.pro_seq_len = int(pro_seq_len)
        self.bin_size = int(bin_size)

        self.feats_order = list(feats_order)
        self.num_feats = len(feats_order)
        self.feats_config = json.load(open(feats_config))
        self.chrom_bins = {
                chrom: (length // bin_size) for chrom, length in hg19_chromsize.items()
                }
        
        self.samples = list()
        self.feats = dict()

        if "_location" in self.feats_config:
            location =self.feats_config["_location"] 
            del self.feats_config["_location"]
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)
        else:
            location = os.path.dirname(os.path.abspath(feats_config))
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)
        
        self.load_datasets()

    def load_datasets(self):
            fn1 = self.enh_datasets
            fn2 = self.pro_datasets
            with open(fn1, 'r') as file1, open(fn2, 'r') as file2:

                for row1, row2 in zip(file1, file2):
                    row1 = row1.strip().split('\t')
                    row2 = row2.strip().split('\t')

                    enh_chrom = row1[1]
                    enh_end = row1[3]
                    enh_start = row1[2]
                    label = row1[0]


                    pro_chrom = row2[1]
                    pro_end = row2[3]
                    pro_start = row2[2]

                    if "all" in fn1:
                        cell = row1[-1]
                    else:       
                        cell = self.cell

                    enh_coord = (int(float(enh_start)) + int(float(enh_end))) // 2
                    tss_coord = (int(float(pro_start)) + int(float(pro_end))) // 2

                    enh_begin = enh_coord - self.enh_seq_len // 2
                    enh_last = enh_coord + self.enh_seq_len // 2

                    pro_begin = tss_coord - self.pro_seq_len // 2
                    pro_last = tss_coord + self.pro_seq_len // 2

                    enh_start_bin, enh_end_bin = enh_begin // self.bin_size, enh_last // self.bin_size
                    pro_start_bin, pro_end_bin = pro_begin // self.bin_size, pro_last // self.bin_size

                    if enh_start_bin < 0:
                        enh_start_bin = 0
                    if enh_end_bin > self.chrom_bins[enh_chrom]: 
                        enh_end_bin = self.chrom_bins[enh_chrom]

                    if pro_start_bin < 0:
                        pro_start_bin = 0
                    if pro_end_bin > self.chrom_bins[pro_chrom]: 
                        pro_end_bin = self.chrom_bins[pro_chrom]
                    

                    self.samples.append((
                        enh_start_bin, enh_end_bin, 
                        pro_start_bin, pro_end_bin, 
                        enh_chrom, pro_chrom, 
                        cell, int(label)
                    ))

                    if cell not in self.feats:
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            self.feats[cell][feat] = torch.load(self.feats_config[cell][feat])
                    
                    


    def __len__(self):
        return len(self.samples)
                    
    def __getitem__(self, idx):
       enh_start_bin, enh_end_bin, pro_start_bin, pro_end_bin, enh_chrom, pro_chrom, cell, label = self.samples[idx]
       enh_ar = torch.zeros((0, enh_end_bin - enh_start_bin))
       pro_ar = torch.zeros((0, pro_end_bin - pro_start_bin))

       for feat in self.feats_order:
           enh_ar = torch.cat((enh_ar, self.feats[cell][feat][enh_chrom][enh_start_bin:enh_end_bin].view(1, -1)), dim=0)
           pro_ar = torch.cat((pro_ar, self.feats[cell][feat][pro_chrom][pro_start_bin:pro_end_bin].view(1, -1)), dim=0)
        
       
       flat_tensor1 = torch.flatten(enh_ar)
       flat_tensor2 = torch.flatten(pro_ar)
       res = torch.cat((flat_tensor1, flat_tensor2), dim=0)

       return res
    
    
    
    def count_lines_in_txt(self, file_path):
        with open(file_path, "r") as file:
            line_count = len(file.readlines())
        
        return line_count
    