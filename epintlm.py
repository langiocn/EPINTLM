from torch import nn
import torch
import numpy as np



NUMBER_WORDS = 4097
NUMBER_POS = 70


EMBEDDING_DIM = 1280
CNN_KERNEL_SIZE = 40
POOL_KERNEL_SIZE = 20
OUT_CHANNELs = 64
LEARNING_RATE = 1e-3


embedding_matrix = torch.tensor(np.load('nucleotide_transformer_6_kmer_embedding.npy'), dtype=torch.float32)
print(embedding_matrix.shape)
print(np.isnan(embedding_matrix).any())
class EPINTLM(nn.Module):
    def __init__(self):
        super(EPINTLM, self).__init__()
        self.embedding_en = nn.Embedding(4097, 1280)
        self.embedding_pr = nn.Embedding(4097, 1280)

        self.embedding_en.weight = nn.Parameter(embedding_matrix)
        self.embedding_pr.weight = nn.Parameter(embedding_matrix)

        self.embedding_en.requires_grad = False
        self.embedding_pr.requires_grad = False

        
        self.enhancer_sequential = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        self.promoter_sequential = nn.Sequential(nn.Conv1d(in_channels=1280, out_channels=64, kernel_size=CNN_KERNEL_SIZE),
                                                 nn.ReLU(),
                                                 nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
                                                 nn.BatchNorm1d(64),
                                                 nn.Dropout(p=0.5)
                                                 )
        

        self.linear_layer = nn.Linear(55, 64)
        self.l1GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)
        self.l2GRU = nn.GRU(input_size=64, hidden_size=32, bidirectional=True, num_layers=2)

        self.self_attn_en = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.self_attn_pr = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        self.self_attn_cr_en = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.self_attn_cr_pr = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        self.self_attn_gen= nn.MultiheadAttention(embed_dim=64, num_heads=8)

       
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.batchnorm1d = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(nn.Linear(245 * 64, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(128, 1),           
               )
   

        self.criterion = nn.BCELoss()
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
        
        
    def forward(self, enhancer_ids, promoter_ids, gene_data):
        SAMPLE_SIZE = enhancer_ids.size(0)

        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)

        gene_data = self.linear_layer(gene_data)
        gene_data = torch.unsqueeze(gene_data, dim=0)
 

        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))

        
        enh_out, _ = self.l1GRU(enhancers_output.permute(2, 0, 1))
        prom_out, _ = self.l2GRU(promoters_output.permute(2, 0, 1))
        

        enhancers_output, _ = self.self_attn_en(enh_out, enh_out, enh_out)
        promoters_output, _ = self.self_attn_cr_pr(prom_out, prom_out, prom_out)

        enh_cr_output, _ = self.self_attn_cr_en(enhancers_output, promoters_output, promoters_output)
        prom_cr_output, _ = self.self_attn_cr_pr(promoters_output, enhancers_output, enhancers_output)

        enh_out = enh_out + enhancers_output + enh_cr_output  # tổng hợp thông tin
        prom_out = prom_out + promoters_output + prom_cr_output

        gene_data, _ = self.self_attn_gen(gene_data, gene_data, gene_data)

        stacked_tensor = torch.cat((enh_out, prom_out), dim=0)
        stacked_tensor = torch.cat((stacked_tensor, gene_data), dim=0).permute(1, 2, 0)
        output = self.batchnorm1d(stacked_tensor)
        output = self.dropout(output)


        result = self.fc(output.flatten(start_dim=1))
       
        return torch.sigmoid(result), output.flatten(start_dim=1)