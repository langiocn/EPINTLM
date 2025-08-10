80% of storage used … If you run out of space, you can't save to Drive, back up Google Photos, or use Gmail.
"""
Improved EPI Model Training Script - Simplified Version
Cải tiến chính:
1. Thêm Early Stopping (patience=20)
2. Cải thiện memory management
3. Better logging và tracking metrics
4. Tối ưu hóa DataLoader
5. Giữ nguyên attention monitoring đơn giản
6. Lưu heatmap của attention weights
7. Không show plot, chỉ save file
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR
from L5.mymodel5 import EPIModel
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from process_data_3 import CombinedDataset
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import random
import gc
from datetime import datetime

# ============= CONFIGURATION =============
class Config:
    """Simple configuration class"""
    def __init__(self):
        # Training parameters
        self.batch_size = 256
        self.num_epochs = 40
        self.pre_train_epoch = 10
        self.learning_rate = 1e-3
        self.weight_decay = 0.001
        self.fine_tune_lr = 1e-4
        self.unfreeze_epoch = 2
        
        # Early stopping - INCREASED TO 20
        self.early_stop_patience = 40
        self.early_stop_min_delta = 1e-4
        
        # Paths
        self.checkpoint_dir = "./checkpoints/"
        self.best_checkpoint_dir = "./best_checkpoints/"
        self.attention_dir = "./attention_analysis/"
        self.data_path = './data/nu_HUVEC_combined_dataset.pt'
        
        # Monitoring frequency
        self.plot_freq = 5
        self.save_freq = 5
        
        # Seed
        self.seed = int(time.time())
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_dirs(self):
        """Create necessary directories"""
        for dir_path in [self.checkpoint_dir, self.best_checkpoint_dir, self.attention_dir]:
            os.makedirs(dir_path, exist_ok=True)


# ============= EARLY STOPPING =============
class EarlyStopping:
    """Simple early stopping implementation"""
    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print('Early stopping triggered!')
        else:
            self.best_score = val_score
            self.counter = 0


# ============= ENHANCED ATTENTION MONITOR =============
class AttentionMonitor:
    """Attention monitoring with heatmap visualization"""
    
    def __init__(self, save_dir="attention_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.attention_stats = {}
        self.sample_weights = {}
        self.epoch_metrics = []
        
    def record_attention(self, attention_dict, epoch, batch_idx=0):
        """Record basic attention statistics"""
        stats = {}
        
        for key in attention_dict:
            if attention_dict[key] is None:
                continue
                
            attn = attention_dict[key].detach().cpu()
            
            # Skip if empty
            if attn.numel() == 0:
                continue
            
            # Basic statistics only - avoid quantile() for large tensors
            stats[key] = {
                'mean': float(attn.mean()),
                'std': float(attn.std()),
                'max': float(attn.max()),
                'min': float(attn.min()),
                'sparsity': float((attn < 0.01).float().mean()),
                'shape': str(attn.shape)
            }
            
            # Save sample for visualization (first batch only)
            if batch_idx == 0:
                if key not in self.sample_weights:
                    self.sample_weights[key] = []
                
                # Take first sample in batch
                sample = attn[0] if attn.dim() > 0 else attn
                self.sample_weights[key].append({
                    'epoch': epoch,
                    'weight': sample.numpy()
                })
        
        return stats
    
    def update_epoch_stats(self, epoch, stats):
        """Save stats for epoch"""
        epoch_data = {'epoch': epoch}
        for key, stat_dict in stats.items():
            for stat_name, value in stat_dict.items():
                if stat_name != 'shape':  # Skip shape in metrics
                    epoch_data[f"{key}_{stat_name}"] = value
        
        self.epoch_metrics.append(epoch_data)
    
    def plot_attention_evolution(self):
        """Plot attention metrics over time - SAVE ONLY, NO SHOW"""
        if not self.epoch_metrics:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.epoch_metrics)
        
        # Get attention types from columns
        attention_types = set()
        for col in df.columns:
            if col != 'epoch' and '_' in col:
                attention_types.add(col.rsplit('_', 1)[0])
        
        attention_types = list(attention_types)
        n_types = len(attention_types)
        
        if n_types == 0:
            return
        
        fig, axes = plt.subplots(n_types, 1, figsize=(12, 4*n_types))
        if n_types == 1:
            axes = [axes]
        
        for idx, attn_type in enumerate(attention_types):
            ax = axes[idx]
            
            # Plot available metrics
            for metric in ['mean', 'std', 'sparsity']:
                col_name = f"{attn_type}_{metric}"
                if col_name in df.columns:
                    ax.plot(df['epoch'], df[col_name], label=metric, marker='o', markersize=3)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(f'{attn_type} Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'attention_evolution.png')
        plt.savefig(save_path, dpi=150)
        plt.close()  # Close figure instead of show
        print(f"Attention evolution saved to: {save_path}")
    
    def plot_attention_heatmaps(self, epoch=None):
        """Generate and save attention heatmaps"""
        if not self.sample_weights:
            print("No attention weights to plot")
            return
        
        for attn_type, weights_list in self.sample_weights.items():
            if not weights_list:
                continue
            
            # Select epochs to visualize
            n_samples = len(weights_list)
            if n_samples >= 4:
                # Show first, 1/3, 2/3, and last epoch
                indices = [0, n_samples//3, 2*n_samples//3, n_samples-1]
            else:
                indices = list(range(n_samples))
            
            fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 5))
            if len(indices) == 1:
                axes = [axes]
            
            fig.suptitle(f'{attn_type} Attention Heatmaps', fontsize=14)
            
            for idx, sample_idx in enumerate(indices):
                weight_data = weights_list[sample_idx]
                weight = weight_data['weight']
                epoch_num = weight_data['epoch']
                
                # Handle different weight dimensions
                if weight.ndim > 2:
                    # If multi-head, average across heads
                    weight = weight.mean(axis=0)
                elif weight.ndim == 1:
                    # If 1D, reshape to 2D for visualization
                    sqrt_len = int(np.sqrt(len(weight)))
                    if sqrt_len * sqrt_len == len(weight):
                        weight = weight.reshape(sqrt_len, sqrt_len)
                    else:
                        # Reshape to narrow matrix if not perfect square
                        weight = weight.reshape(-1, 1)
                
                # Create heatmap
                im = axes[idx].imshow(weight, cmap='hot', aspect='auto', interpolation='nearest')
                axes[idx].set_title(f'Epoch {epoch_num}')
                axes[idx].set_xlabel('Position')
                axes[idx].set_ylabel('Position')
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f'{attn_type}_heatmaps.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Heatmap saved to: {save_path}")
    
    def save_summary(self):
        """Save summary to JSON"""
        summary = {
            'total_epochs': len(self.epoch_metrics),
            'metrics': self.epoch_metrics
        }
        
        with open(os.path.join(self.save_dir, 'attention_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Attention summary saved to {self.save_dir}")


# ============= TRAINING METRICS TRACKER =============
class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_aupr': [],
            'val_auc': [],
            'learning_rate': []
        }
        
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot_curves(self, save_path='training_curves.png'):
        """Plot training curves - SAVE ONLY, NO SHOW"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train', marker='o', markersize=3)
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val', marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUPR
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['val_aupr'], color='green', marker='o', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUPR')
        axes[0, 1].set_title('Validation AUPR')
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['val_auc'], color='orange', marker='o', markersize=3)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.metrics['epoch'], self.metrics['learning_rate'], color='red', marker='o', markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()  # Close figure instead of show
        print(f"📈 Training curves saved to: {save_path}")
    
    def save(self, path='metrics.json'):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


# ============= HELPER FUNCTIONS =============
def set_embedding_requires_grad(model, requires_grad: bool):
    """Control embedding layer gradients"""
    model.embedding_en.weight.requires_grad = requires_grad
    model.embedding_pr.weight.requires_grad = requires_grad


def get_num_correct(preds, labels):
    """Calculate number of correct predictions"""
    predictions = (preds >= 0.5).float()
    return (predictions == labels).sum().item()


def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============= TRAINING FUNCTIONS =============
def train_epoch(model, dataloader, optimizer, device, attention_monitor=None, epoch=0):
    """Train for one epoch with attention monitoring"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_attention_stats = []
    
    for batch_idx, data in enumerate(dataloader):
        enhancer_ids, promoter_ids, gene_data, labels = data
        enhancer_ids = enhancer_ids.to(device)
        promoter_ids = promoter_ids.to(device)
        gene_data = gene_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, attention_dict = model(enhancer_ids, promoter_ids, gene_data)
        
        # Record attention for first batch
        if attention_monitor and batch_idx == 0:
            stats = attention_monitor.record_attention(attention_dict, epoch, batch_idx)
             # Ghi lại attention cho batch đầu tiên mỗi epoch
            for key, stat_dict in stats.items():
                print(f"  {key}:")
                print(f"    Shape: {stat_dict.get('shape', 'N/A')}")
                print(f"    Mean: {stat_dict['mean']:.4f}, std: {stat_dict['std']:.4f}")
                print(f"    Max: {stat_dict['max']:.4f}, Min: {stat_dict['min']:.4f}")
                print(f"    Sparsity: {stat_dict['sparsity']:.4f}")
            
        

            if stats:
                batch_attention_stats.append(stats)
        
        # Prepare labels
        labels = labels.unsqueeze(1).float()
        if labels.shape == torch.Size([1, 1]):
            labels = torch.reshape(labels, (1,))
        
        # Calculate loss
        loss = model.criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()

        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track accuracy
        total_correct += get_num_correct(outputs, labels)
        total_samples += labels.size(0)
        
        # Clear memory periodically
        if batch_idx % 50 == 0:
            clear_memory()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy, batch_attention_stats


def validate(model, dataloader, device, attention_monitor=None, epoch=0):
    """Validation with metrics calculation"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    val_attention_stats = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            enhancer_ids, promoter_ids, gene_data, labels = data
            enhancer_ids = enhancer_ids.to(device)
            promoter_ids = promoter_ids.to(device)
            gene_data = gene_data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, attention_dict = model(enhancer_ids, promoter_ids, gene_data)
            
            # Record attention for first batch
            if attention_monitor and batch_idx == 0:
                stats = attention_monitor.record_attention(attention_dict, epoch, batch_idx)
                if stats:
                    val_attention_stats.append(stats)
            
            # Prepare labels
            labels = labels.unsqueeze(1).float()
            if labels.shape == torch.Size([1, 1]):
                labels = torch.reshape(labels, (1,))
            
            # Calculate loss
            loss = model.criterion(outputs, labels)
            total_loss += loss.item()
            
            # Collect predictions
            all_preds.extend(outputs.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    aupr = average_precision_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    # Check prediction diversity
    pred_binary = (np.array(all_preds) >= 0.5).astype(int)
    unique_preds = np.unique(pred_binary)
    pos_ratio = np.mean(pred_binary)
    
    print(f"  Unique predictions: {unique_preds}, Positive ratio: {pos_ratio:.2%}")
    
    return avg_loss, aupr, auc, val_attention_stats


# ============= DATA PREPARATION =============
def prepare_data(config):
    """Prepare data loaders"""
    # Load dataset
    torch.serialization.add_safe_globals([CombinedDataset])
    dataset = torch.load(config.data_path, weights_only=False)
    
    # Create enhancer-based split
    enhancer_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        enhancer_str = ''.join(map(str, dataset[i][0].tolist()))
        enhancer_to_indices[enhancer_str].append(i)
    
    unique_enhancers = list(enhancer_to_indices.keys())
    enhancer_labels = [dataset[enhancer_to_indices[enh][0]][3].item() 
                      for enh in unique_enhancers]
    
    # Split data
    train_enh, test_enh = train_test_split(
        unique_enhancers,
        test_size=0.1,
        stratify=enhancer_labels,
        random_state=config.seed
    )
    
    train_idx = [i for enh in train_enh for i in enhancer_to_indices[enh]]
    test_idx = [i for enh in test_enh for i in enhancer_to_indices[enh]]
    
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Check distribution
    train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][3].item() for i in range(len(test_dataset))]
    
    print(f"Train: {len(train_dataset)} samples, Positive: {sum(train_labels)/len(train_labels):.2%}")
    print(f"Test: {len(test_dataset)} samples, Positive: {sum(test_labels)/len(test_labels):.2%}")
    
    # Create loaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============= MAIN TRAINING =============
def main():
    """Main training function"""
    
    # Setup
    config = Config()
    config.create_dirs()
    
    print("="*60)
    print(f"Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"Early Stopping Patience: {config.early_stop_patience}")
    print("="*60)
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Initialize model
    model = EPIModel()
    model.to(config.device)
    
    # Check GPU
    if config.device.type == 'cuda':
        print(f"Model on GPU: {all(p.is_cuda for p in model.parameters())}")
    
    # Initialize monitoring
    attention_monitor = AttentionMonitor(config.attention_dir)
    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta
    )
    
    # Prepare data
    train_loader, val_loader = prepare_data(config)
    
    # Initially freeze embeddings
    # set_embedding_requires_grad(model, False)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
    
    # Training loop
    best_aupr = 0
    best_epoch = 0
    best_val_loss = 9999
    
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.num_epochs-1}")
        print(f"{'='*50}")
        
        # Unfreeze embeddings at specified epoch
        if epoch == config.unfreeze_epoch:
            print("Unfreezing embeddings for fine-tuning...")
            set_embedding_requires_grad(model, True)
            
            # Recreate optimizer with all parameters
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.fine_tune_lr,
                weight_decay=config.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.1)
        
        # Training
        train_loss, train_acc, train_attention_stats = train_epoch(
            model, train_loader, optimizer, config.device, attention_monitor, epoch
        )
        
        # Validation
        val_loss, val_aupr, val_auc, val_attention_stats = validate(
            model, val_loader, config.device, attention_monitor, epoch
        )
        
        # Update attention stats
        if train_attention_stats:
            for stats in train_attention_stats:
                attention_monitor.update_epoch_stats(epoch, stats)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, AUPR: {val_aupr:.4f}, AUC: {val_auc:.4f}")
        print(f"Learning rate: {current_lr:.6f}")
        
        # Track metrics
        metrics_tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_aupr=val_aupr,
            val_auc=val_auc,
            learning_rate=current_lr
        )
        
        # Save checkpoint
        if (epoch + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if (val_aupr > best_aupr) or (val_aupr == best_aupr and val_loss < best_val_loss):
            best_aupr = val_aupr
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(config.best_checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_aupr': val_aupr,
                'val_auc': val_auc
            }, best_path)
            print(f"New best model! AUPR: {best_aupr:.4f}")
        
        # Early stopping check
        early_stopping(val_aupr)
        if early_stopping.early_stop:
            break
        
        # Generate plots
        if (epoch + 1) % config.plot_freq == 0:
            print("Generating plots...")
            attention_monitor.plot_attention_evolution()
            attention_monitor.plot_attention_heatmaps(epoch)
            metrics_tracker.plot_curves(os.path.join(config.attention_dir, f'training_curves_epoch_{epoch}.png'))
        
        # Clear memory
        clear_memory()
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
    # Generate final plots
    print("Generating final analysis...")
    attention_monitor.plot_attention_evolution()
    attention_monitor.plot_attention_heatmaps()
    attention_monitor.save_summary()
    metrics_tracker.plot_curves(os.path.join(config.attention_dir, 'final_training_curves.png'))
    metrics_tracker.save(os.path.join(config.attention_dir, 'training_metrics.json'))
    
    print(f"Best AUPR: {best_aupr:.4f} at epoch {best_epoch}")
    print(f"Models saved in: {config.checkpoint_dir}")
    print(f"Best model saved in: {config.best_checkpoint_dir}")
    print(f"Analysis saved in: {config.attention_dir}")
    
    return model, best_aupr


# ============= RUN TRAINING =============
if __name__ == "__main__":
    model, best_aupr = main()