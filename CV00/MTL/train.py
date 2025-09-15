# src/train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import numpy as np

# Import our custom modules
from models.aifr import AIFRModel
from losses.age_loss import AgeLoss
from losses.cosface_loss import CosFaceLoss
# # Assuming your FaceDataset class is in data.dataloader
from data.data_loader import FaceDataset, train_augmentations, val_augmentations, mtcnn_processor

# ANSI color codes for prettier console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])

        self._setup_dataloaders()

        self.model = AIFRModel(
            backbone_name=config['model']['backbone_name'],
            embedding_size=config['model']['embedding_size'],
            num_age_bins=config['model']['num_age_bins'],
            pretrained=config['model']['pretrained']
        ).to(self.device)

        num_train_identities = self.train_dataset.manifest['identity_id'].nunique()
        self.cosface_loss = CosFaceLoss(
            embedding_size=config['model']['embedding_size'],
            num_classes=num_train_identities
        ).to(self.device)
        self.age_loss = AgeLoss().to(self.device)

        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.cosface_loss.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['training']['epochs'])
        self.scaler = torch.cuda.amp.GradScaler()

        self.best_val_mae = float('inf')
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    def _setup_dataloaders(self):
        # ... (This method remains unchanged from the previous version) ...
        print("Setting up dataloaders...")
        self.train_dataset = FaceDataset(
            manifest_path=self.config['data']['train_manifest'],
            target_count_per_class=self.config['data']['target_count_per_class'],
            mtcnn_preprocessor=mtcnn_processor, # Assuming mtcnn_processor is initialized globally
            augmentations=train_augmentations,
            is_train=True
        )
        self.val_dataset = FaceDataset(
            manifest_path=self.config['data']['val_manifest'],
            target_count_per_class=None,
            mtcnn_preprocessor=mtcnn_processor,
            augmentations=val_augmentations,
            is_train=False
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config['training']['batch_size'],
            shuffle=True, num_workers=self.config['data']['num_workers'], pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config['training']['batch_size'],
            shuffle=False, num_workers=self.config['data']['num_workers'], pin_memory=True
        )
        print("Dataloaders are ready.")

    def _get_age_metrics(self, age_logits, true_ages, true_age_groups):
        # Calculate MAE
        softmax_probs = F.softmax(age_logits, dim=1)
        age_range = torch.arange(0, self.config['model']['num_age_bins'], device=self.device)
        predicted_ages = (softmax_probs * age_range).sum(dim=1)
        mae = F.l1_loss(predicted_ages, true_ages).item()

        # Calculate Group Accuracy
        group_ranges = self.age_loss.group_ranges
        group_logits = torch.stack([age_logits[:, start:end].sum(dim=1) for start, end in group_ranges], dim=1)
        predicted_groups = torch.argmax(group_logits, dim=1)
        group_acc = (predicted_groups == true_age_groups).float().mean().item()

        return mae, group_acc

    def _train_one_epoch(self, lambda_grl):
        self.model.train()
        metrics = {'loss': 0, 'loss_cosface': 0, 'loss_age': 0, 'loss_adv': 0, 'age_mae': 0, 'age_group_acc': 0}

        progress_bar = tqdm(self.train_loader, desc=f"{Colors.OKCYAN}Training{Colors.ENDC}", leave=False)
        for batch in progress_bar:
            #print("Start the loop inside train one epoch function: ")
            images, ages, age_groups, identities = [item.to(self.device) for item in batch]


            self.optimizer.zero_grad()
            #print("Calculating the loss: ")
            with torch.cuda.amp.autocast():
                outputs = self.model(images, lambda_grl=lambda_grl)
                # print("outputs: ", outputs)
                # # save the outputs in text file as text not binary:
                # with open('adversarial_age_logits.txt', 'w') as f:
                #     f.write(str(outputs['adversarial_age_logits']))

                # with open('age_logits.txt', 'w') as f:
                #     f.write(str(outputs['age_logits']))


                loss_c = self.cosface_loss(outputs['embedding'], identities)
                loss_a = self.age_loss(outputs['age_logits'], ages, age_groups)
                loss_adv = self.age_loss(outputs['adversarial_age_logits'], ages, age_groups)

                loss = (loss_c +
                        self.config['training']['lambda_age'] * loss_a +
                        self.config['training']['lambda_adv'] * loss_adv)

            #print("Updating the values: ")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            metrics['loss'] += loss.item()
            metrics['loss_cosface'] += loss_c.item()
            metrics['loss_age'] += loss_a.item()
            metrics['loss_adv'] += loss_adv.item()

            mae, group_acc = self._get_age_metrics(outputs['age_logits'], ages, age_groups)
            metrics['age_mae'] += mae
            metrics['age_group_acc'] += group_acc

            progress_bar.set_postfix(loss=f"{loss.item():.3f}", mae=f"{mae:.2f}")

        # Average metrics over all batches
        for k in metrics:
            metrics[k] /= len(self.train_loader)
        return metrics

    def _validate_one_epoch(self):
        self.model.eval()
        metrics = {'loss': 0, 'loss_cosface': 0, 'loss_age': 0, 'age_mae': 0, 'age_group_acc': 0}
        all_embeddings, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"{Colors.OKBLUE}Validating{Colors.ENDC}", leave=False):
                images, ages, age_groups, identities = [item.to(self.device) for item in batch]

                outputs = self.model(images, lambda_grl=0.0)

                loss_c = self.cosface_loss(outputs['embedding'], identities)
                loss_a = self.age_loss(outputs['age_logits'], ages, age_groups)
                loss = loss_c + self.config['training']['lambda_age'] * loss_a

                metrics['loss'] += loss.item()
                metrics['loss_cosface'] += loss_c.item()
                metrics['loss_age'] += loss_a.item()

                mae, group_acc = self._get_age_metrics(outputs['age_logits'], ages, age_groups)
                metrics['age_mae'] += mae
                metrics['age_group_acc'] += group_acc

                all_embeddings.append(outputs['embedding'].cpu())
                all_labels.append(identities.cpu())

        for k in metrics:
            metrics[k] /= len(self.val_loader)

        # Calculate embedding quality
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        # To avoid OOM on large val sets, we sample a subset for similarity calculation
        sample_indices = np.random.choice(len(all_embeddings), min(2048, len(all_embeddings)), replace=False)
        sample_embeddings = all_embeddings[sample_indices]
        sample_labels = all_labels[sample_indices]

        sim_matrix = F.cosine_similarity(sample_embeddings.unsqueeze(1), sample_embeddings.unsqueeze(0), dim=2)
        label_matrix = sample_labels.unsqueeze(1) == sample_labels.unsqueeze(0)
        # Remove diagonal (self-similarity)
        sim_matrix.fill_diagonal_(float('-inf'))
        label_matrix.fill_diagonal_(False)

        if label_matrix.any(): # Ensure there are positive pairs in the sample
            metrics['intra_class_sim'] = sim_matrix[label_matrix].mean().item()
        else:
            metrics['intra_class_sim'] = 0.0

        metrics['inter_class_sim'] = sim_matrix[~label_matrix].mean().item()

        return metrics

    def train(self):
        print(f"{Colors.HEADER}{Colors.BOLD}Starting AIFR Model Training{Colors.ENDC}")
        for epoch in range(self.config['training']['epochs']):
            print(f"\n{Colors.BOLD}--- Epoch {epoch+1}/{self.config['training']['epochs']} ---{Colors.ENDC}")

            current_lambda_grl = min(1.0, (epoch / self.config['training']['epochs']) * 2)
            #print("Calling train one epoch function(): ")
            train_metrics = self._train_one_epoch(lambda_grl=current_lambda_grl)
            #print("Calling validate one epoch function(): ")
            val_metrics = self._validate_one_epoch()
            self.scheduler.step()

            # --- Rich Logging Table ---
            print(f"{Colors.UNDERLINE}" + "-"*85 + f"{Colors.ENDC}")
            print(f"| {'Stage':<10} | {'Total Loss':<10} | {'CosFace Loss':<12} | {'Age Loss':<10} | {'Age MAE':<10} | {'Group Acc':<10} |")
            print("-"*85)
            print(f"| {'Train':<10} | {train_metrics['loss']:<10.4f} | {train_metrics['loss_cosface']:<12.4f} | {train_metrics['loss_age']:<10.4f} | {train_metrics['age_mae']:<10.2f} | {train_metrics['age_group_acc']*100:<9.2f}% |")
            print(f"| {'Validation':<10} | {val_metrics['loss']:<10.4f} | {val_metrics['loss_cosface']:<12.4f} | {val_metrics['loss_age']:<10.4f} | {Colors.BOLD}{val_metrics['age_mae']:<10.2f}{Colors.ENDC} | {val_metrics['age_group_acc']*100:<9.2f}% |")
            print("-"*85)
            print(f"Validation Embedding Quality: Intra-Class Sim: {Colors.OKGREEN}{val_metrics['intra_class_sim']:.3f}{Colors.ENDC} | Inter-Class Sim: {Colors.FAIL}{val_metrics['inter_class_sim']:.3f}{Colors.ENDC}")

            val_mae = val_metrics['age_mae']
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                save_path = os.path.join(self.config['training']['checkpoint_dir'], "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"{Colors.OKGREEN}🎉 New best model saved with MAE: {val_mae:.2f} at {save_path}{Colors.ENDC}")





if __name__ == '__main__':
    CONFIG = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': { 'train_manifest': 'manifests/train_manifest.csv', 'val_manifest': 'manifests/val_manifest.csv', 'target_count_per_class': 90, 'num_workers': 0 },
        'model': { 'backbone_name': 'resnet50', 'embedding_size': 512, 'num_age_bins': 101, 'pretrained': True },
        'training': { 'epochs': 50, 'batch_size': 64, 'learning_rate': 1e-4, 'weight_decay': 5e-4, 'lambda_age': 0.5, 'lambda_adv': 0.7, 'checkpoint_dir': 'checkpoints/' }
    }

    trainer = Trainer(CONFIG)
    trainer.train()
