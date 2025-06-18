from torch.utils.data import DataLoader
from epintlm import EPINTLM
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
import torch
import numpy as np
from seqgendataset import SeqGenDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import defaultdict


import random
import time

# Lấy thời gian hiện tại làm seed
seed_value = int(time.time())  # Thời gian hiện tại (timestamp)
print(f"Using seed: {seed_value}")

# Cài đặt seed cho numpy, random và pytorch
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)

batch_size = 256
epoch = 60
pre_train_epoch = 10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_embedding_requires_grad(model, requires_grad: bool):
    model.embedding_en.weight.requires_grad = requires_grad
    model.embedding_pr.weight.requires_grad = requires_grad

def check_performance(model, dataloader):
    model.eval()
    test_epoch_loss = 0.0
    test_epoch_correct = 0
    test_epoch_preds = torch.tensor([]).to(device)
    test_epoch_target = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in dataloader:
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            gene_data = gene_data.to(device)
            labels = labels.to(device)
                
                
            outputs, _ = model(enhancer_ids, promoter_ids, gene_data)
            labels = labels.unsqueeze(1).float()
            test_epoch_target = torch.cat((test_epoch_target, labels.view(-1)))

            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
                
            loss = model.criterion(outputs, labels)
            test_epoch_preds = torch.cat((test_epoch_preds, outputs.view(-1)))
            test_epoch_loss += loss.item()
            test_epoch_correct += get_true_labels(outputs, labels)

    test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    
    pred_labels = (test_epoch_preds >= 0.5).float().cpu().numpy()
    true_labels = test_epoch_target.cpu().numpy()

    unique_preds = np.unique(pred_labels)
    num_positive_preds = int(np.sum(pred_labels))
    total_true = int(np.sum(true_labels))

    print(f"Unique predicted labels: {unique_preds}")
    print(f"Predicted 1s: {num_positive_preds}/{total_true} ({(num_positive_preds/total_true)*100:.2f}%)")

    return test_epoch_loss, test_epoch_aupr, test_epoch_auc

def get_true_labels(preds, labels):
    predictions = (preds >= 0.5).float()
    
    return (predictions == labels).sum().item()


file_path = "./checkpoints/"
file_path_best = "./best_checkpoints/"
epimodel = EPINTLM()
epimodel.to(device)

optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, epimodel.parameters()),
        lr=1e-3,
        weight_decay=0.001
    )

scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)

all_params_on_gpu = all(param.is_cuda for param in epimodel.parameters())
print("cuda: ",torch.cuda.is_available())
print("gpu params:",all_params_on_gpu)

    

#LOAD TRAINING DATA

torch.serialization.add_safe_globals([SeqGenDataset])
combined_dataset = torch.load('./data/nu_HUVEC_combined_dataset.pt', weights_only=False)

def enhancer_to_str(tensor):
    return ''.join(map(str, tensor.tolist()))

enhancer_to_indices = defaultdict(list)
labels = []

for i in range(len(combined_dataset)):
    enhancer_str = enhancer_to_str(combined_dataset[i][0])  # enhancer tensor ở vị trí 0
    enhancer_to_indices[enhancer_str].append(i)

unique_enhancers = list(enhancer_to_indices.keys())
enhancer_labels = [combined_dataset[enhancer_to_indices[enh][0]][3].item() for enh in unique_enhancers]

train_enh, test_enh = train_test_split(
    unique_enhancers,
    test_size=0.1,
    stratify=enhancer_labels,
    random_state=42
)

train_idx = [i for enh in train_enh for i in enhancer_to_indices[enh]]
test_idx = [i for enh in test_enh for i in enhancer_to_indices[enh]]

train_dataset = Subset(combined_dataset, train_idx)
test_dataset = Subset(combined_dataset, test_idx)

def check_label_distribution(dataset, name=""):
    labels = [dataset[i][3].item() for i in range(len(dataset))]
    ratio = sum(labels) / len(labels)
    print(f"{name} size: {len(dataset)} | Label 1 ratio: {ratio:.2f}")

check_label_distribution(train_dataset, "Train")
check_label_distribution(test_dataset, "Test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

torch.cuda.empty_cache()

best_aupr = 0
for i in range(epoch):
        print(f'Epoch: {i}')
        if i == 2:
            print("Unfreezing embeddings for fine-tuning...")
            set_embedding_requires_grad(epimodel, True)

            optimizer = torch.optim.Adam(epimodel.parameters(), 
                lr=1e-4, 
                weight_decay=0.001)
            scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            
        epimodel.train()
        train_epoch_loss = 0.0
        train_epoch_correct = 0
        print('Load data in train_loader')
        for data in train_loader:
            optimizer.zero_grad()
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            gene_data = gene_data.to(device)
            promoter_ids = promoter_ids.to(device)
            labels = labels.to(device)

            outputs, emd = epimodel(enhancer_ids, promoter_ids, gene_data)

            labels = labels.unsqueeze(1).float()
            
            
            if labels.shape == torch.Size([1, 1]):
              labels = torch.reshape(labels, (1,))

            loss = epimodel.criterion(outputs, labels)
            train_epoch_loss += loss.item()
          
            loss.backward()
            optimizer.step()
            
            
        avgloss = train_epoch_loss / len(train_loader)
        scheduler.step()
        print("learning_rate:", scheduler.get_last_lr())
        test_epoch_loss, test_epoch_aupr, test_epoch_auc = check_performance(epimodel, val_loader)
        print(f'Average loss per batch: {avgloss:.4f}, aupr: {test_epoch_aupr}, auc: {test_epoch_auc}')
        torch.save(epimodel.state_dict(), file_path + 'NU_nocross_huvec_train_model_' + str(i) + '.pt')
        if test_epoch_aupr >= best_aupr:
            best_aupr = test_epoch_aupr
            torch.save(epimodel.state_dict(), file_path_best + 'NU_nocross_huvec_best.pt')
            print(f"[Epoch {i}] New best AUPR: {best_aupr:.4f} - Model Saved!")
        