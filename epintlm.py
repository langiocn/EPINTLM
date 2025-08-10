
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


class ImprovedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, need_weights=True):
        q_norm = self.norm(q)
        k_norm = self.norm(k)
        v_norm = self.norm(v)
        attn_out, attn_weights = self.attn(q_norm, k_norm, v_norm, need_weights=need_weights, average_attn_weights = False)
        return self.dropout(attn_out), attn_weights
    
class EPIModel(nn.Module):
    def __init__(self):
        super(EPIModel, self).__init__()

        #embedding
        self.embedding_en = nn.Embedding(4097, 1280)
        self.embedding_pr = nn.Embedding(4097, 1280)

        self.embedding_en.weight = nn.Parameter(embedding_matrix, requires_grad = False)
        self.embedding_pr.weight = nn.Parameter(embedding_matrix, requires_grad = False)

        # self.embedding_en.weight.requires_grad = False
        # self.embedding_pr.requires_grad = False

        
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

        self.self_attn_en = ImprovedAttention(embed_dim=64, num_heads=8)
        self.self_attn_pr = ImprovedAttention(embed_dim=64, num_heads=8)

        self.self_attn_cr_en = ImprovedAttention(embed_dim=64, num_heads=8)
        self.self_attn_cr_pr = ImprovedAttention(embed_dim=64, num_heads=8)

        self.self_attn_gen= ImprovedAttention(embed_dim=64, num_heads=8)


        
        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.batchnorm1d = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(nn.Linear(245 * 64, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(128, 1)      
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

        enhancer_embedding = self.embedding_en(enhancer_ids)
        promoter_embedding = self.embedding_pr(promoter_ids)

        gene_data = self.linear_layer(gene_data)
        gene_data = torch.unsqueeze(gene_data, dim=0)
 

        enhancers_output = self.enhancer_sequential(enhancer_embedding.permute(0, 2, 1))
        promoters_output = self.promoter_sequential(promoter_embedding.permute(0, 2, 1))

        
        enh_out, _ = self.l1GRU(enhancers_output.permute(2, 0, 1))
        prom_out, _ = self.l2GRU(promoters_output.permute(2, 0, 1))
        

        enhancers_output, self_enh_weights = self.self_attn_en(enh_out, enh_out, enh_out)
        promoters_output, self_prom_weights = self.self_attn_pr(prom_out, prom_out, prom_out)

        enh_cr_output, enh_cross_attn_weights = self.self_attn_cr_en(enh_out, prom_out, prom_out)
        prom_cr_output, prom_cross_attn_weights = self.self_attn_cr_pr(prom_out, enh_out, enh_out)

        enh_out = enh_out + enhancers_output + enh_cr_output 
        prom_out = prom_out + promoters_output + prom_cr_output

        gene_data, gen_attn_weights = self.self_attn_gen(gene_data, gene_data, gene_data)

        stacked_tensor = torch.cat((enh_out, prom_out), dim=0)
        stacked_tensor = torch.cat((stacked_tensor, gene_data), dim=0).permute(1, 2, 0)
        output = self.batchnorm1d(stacked_tensor)
        output = self.dropout(output)
   
        
      
        result = self.fc(output.flatten(start_dim=1))
       
        attention_dict = {
            'cross_attn_enhancer': enh_cross_attn_weights,
            'cross_attn_promoter': prom_cross_attn_weights,
            'self_enh_weights': self_enh_weights,
            'self_prom_weights': self_prom_weights,
            'gen_attn': gen_attn_weights
        }
        
        
        return torch.sigmoid(result), attention_dict