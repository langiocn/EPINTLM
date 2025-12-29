import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from feature_extraction.seqgendataset import SeqGenDataset
from model.epintlm import EPIModel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
    
# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EPIModel().to(device)

# # Load checkpoint dictionary
checkpoint = torch.load('/content/EPIPDLF/best_checkpoints/best_model.pt', map_location=device, weights_only=False)


# Load model weights
model.load_state_dict(checkpoint['model_state_dict'])

# model = EPIModel().to(device)  # Thay YourModelClass bằng class model bạn đã định nghĩa
# model.load_state_dict(torch.load('/content/EPIPDLF/checkpoints/model_epoch_59.pt', map_location=device))
# model.eval()

# Set to evaluation mode
model.eval()

# LOAD TESTING DATA
batch_size = 512
torch.serialization.add_safe_globals([SeqGenDataset])
test_combined_dataset = torch.load('./data/nu_HUVEC_combined_dataset_test.pt', weights_only=False)
print(f"✅ TEST Dataset length: {len(test_combined_dataset)}")
test_loader = DataLoader(test_combined_dataset, batch_size=batch_size, shuffle=True)

def val_forwrd(model, dataloader):
    model.eval()
    test_epoch_loss = 0.0
    test_epoch_correct = 0
    test_epoch_preds = torch.tensor([]).to(device)
    test_epoch_target = torch.tensor([]).to(device)
    all_attention_outputs = []
    all_labels = []
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
            test_epoch_correct += get_num_correct(outputs, labels)
                
    pred_labels = (test_epoch_preds >= 0.5).float().cpu().detach().numpy()
    true_labels = test_epoch_target.cpu().detach().numpy()
     # ✅ In thông tin kiểm tra đa dạng và số lượng nhãn 1
    unique_preds = np.unique(pred_labels)
    num_positive_preds = int(np.sum(pred_labels))
    total_true = int(np.sum(true_labels))

    print(f"🧠 Unique predicted labels: {unique_preds}")
    print(f"✅ Predicted 1s: {num_positive_preds}/{total_true} ({(num_positive_preds/total_true)*100:.2f}%)")
    
    #visualize_with_tsne(torch.cat(all_attention_outputs, dim=0), torch.cat(all_labels, dim=0), "test_" + str(i) + ".png")
    test_epoch_aupr = average_precision_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    test_epoch_auc = roc_auc_score(test_epoch_target.cpu().detach().numpy(), test_epoch_preds.cpu().detach().numpy())
    pred_labels = (test_epoch_preds >= 0.5).float().cpu().numpy()
    true_labels = test_epoch_target.cpu().numpy()
    test_epoch_f1 = f1_score(true_labels, pred_labels)
    return test_epoch_loss, test_epoch_aupr, test_epoch_auc, test_epoch_f1

def get_num_correct(preds, labels):
    predictions = (preds >= 0.5).float()
    
    return (predictions == labels).sum().item()

# Gọi hàm eval và in ra các metric
val_loss, val_aupr, val_auc, test_epoch_f1 = val_forwrd(model, test_loader)
print(f'Validation loss: {val_loss:.4f}, AUPR: {val_aupr:.4f}, AUC: {val_auc:.4f}, f1: {test_epoch_f1:.4f}')
