import copy
import swanlab
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def monitor_components(model_dict, optimizers):
    """
    Monitor training status of each component.

    Args:
        model_dict: {"bert": bert_model, "trans": trans_model, "fusion": fusion_model}
        optimizers: corresponding optimizers for each component
    """
    print("\n" + "=" * 60)
    for name in ["bert", "trans", "fusion"]:
        model = model_dict[name]
        optim = optimizers[name]

        # Check update-to-weight ratio
        max_ratio = 0
        for param in model.parameters():
            if param.grad is not None:
                ratio = (param.grad.abs() * optim.param_groups[0]['lr']) / (param.abs() + 1e-6)
                max_ratio = max(max_ratio, ratio.max().item())

        # Check optimizer momentum statistics (Adam-style)
        mom_stats = []
        for param in model.parameters():
            if param in optim.state:
                mom_stats.append(optim.state[param]['exp_avg'].abs().mean().item())

        # Optional detailed logging (disabled by default)
        # print(
        #     f"[{name.upper():<10}] "
        #     f"LR: {optim.param_groups[0]['lr']:.1e} | "
        #     f"Update Ratio: {max_ratio:.3e} | "
        #     f"Momentum Mean: {np.mean(mom_stats):.3e} | "
        #     f"Grad Norm: {sum(p.grad.norm() for p in model.parameters() if p.grad is not None):.3e}"
        #     )
    # print("=" * 60)


class ModalityAwareNormalizer:
    def __init__(self, modalities):
        """
        Improved modality-aware gradient normalizer.

        Example input:
        modalities = {
            'text': {
                'params': text_model.parameters(),
                'clip_val': 0.1
            },
            'trans': {
                'params': trans_model.parameters(),
                'clip_val': 1.0
            },
            'fusion': {
                'params': fusion_model.parameters(),
                'clip_val': 0.5
            }
        }
        """
        self.modalities = modalities

    def normalize(self):
        """
        Perform per-modality gradient normalization.
        """
        for mod_name, config in self.modalities.items():
            params = config['params']
            target_norm = config['clip_val']

            # Collect valid gradients
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue

            # Compute current gradient norm for this modality
            current_norm = torch.norm(
                torch.stack([torch.norm(g) for g in grads])
            )

            # Avoid division by zero
            if current_norm < 1e-6:
                print(f"Warning: {mod_name} gradient norm is near zero ({current_norm:.2e})")
                continue

            # Compute scaling factor and apply
            scale = target_norm / current_norm
            for g in grads:
                g.mul_(scale)

            # Optional verification
            # new_norm = torch.norm(
            #     torch.stack([torch.norm(g) for g in grads])
            # )
            # print(f"{mod_name} grad norm: {current_norm:.2e} → {new_norm:.2e}")


class FlowTrainer:
    """
    Trainer for multimodal network traffic classification models.
    """

    def __init__(self, num_classes, text_model, trans_model, fusion_model,
                 text_optim, trans_optim, fusion_optim,
                 text_scheduler, trans_scheduler, fusion_scheduler,
                 device, use_parallel, start_freeze_layer, swanlab):
        self.num_classes = num_classes
        self.text_model = text_model
        self.trans_model = trans_model
        self.fusion_model = fusion_model

        self.text_optim = text_optim
        self.trans_optim = trans_optim
        self.fusion_optim = fusion_optim

        self.text_scheduler = text_scheduler
        self.trans_scheduler = trans_scheduler
        self.fusion_scheduler = fusion_scheduler

        self.lora_warmup_epochs = 5
        self.device = device
        self.use_parallel = use_parallel
        self.current_layer = start_freeze_layer

        self.validation_loss_history = []
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Store best text-only accuracy
        self.text_acc = 0.0
        self.text_acc_threshold = 0.98

        # Cache CLS features during fusion warmup
        self.cls_feat = []

        # SwanLab experiment handler
        self.swanlab = swanlab

    def train_epoch(self, loader, trans_train_loader):
        """
        Train one epoch.

        Switches automatically between:
        - Text-only training (BERT-only)
        - Full multimodal training (Text + Transformer + Fusion)
        """
        # If text-only accuracy is high enough, switch to BERT-only mode
        if self.text_acc > self.text_acc_threshold:
            if self.use_parallel:
                self.trans_model.module.warmup = True
            else:
                self.trans_model.warmup = True

            self.text_model.train()
            total_loss = 0
            all_preds, all_labels = [], []

            for batch_idx, batch1 in enumerate(loader):
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                self.text_optim.zero_grad()

                # Text feature extraction
                bert_output = self.text_model(
                    inputs["input_ids"], inputs["attention_mask"]
                )

                loss = self.criterion(bert_output, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()
                loss.backward()

                # Gradient clipping
                params = self.text_model.module.parameters() if self.use_parallel else self.text_model.parameters()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, params),
                    max_norm=10.0
                )

                self.text_optim.step()
                self.text_scheduler.step()

                total_loss += loss.item()
                all_preds.extend(bert_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="train")

            return metrics, all_preds, all_labels, self.text_model, None, None

        # ================= Full multimodal training =================
        else:
            if self.use_parallel:
                self.trans_model.module.warmup = False
                self.text_model.module.warmup = False
            else:
                self.trans_model.warmup = False
                self.text_model.warmup = False

            self.text_model.train()
            self.trans_model.train()
            self.fusion_model.train()

            total_loss = 0
            all_preds, all_labels = [], []

            # Define modality-wise gradient normalization config
            if self.use_parallel:
                modalities = {
                    'text': {'params': self.text_model.module.parameters(), 'clip_val': 0.1},
                    'trans': {'params': self.trans_model.module.parameters(), 'clip_val': 1.0},
                    'fusion': {'params': self.fusion_model.module.parameters(), 'clip_val': 0.5}
                }
            else:
                modalities = {
                    'text': {'params': self.text_model.parameters(), 'clip_val': 0.1},
                    'trans': {'params': self.trans_model.parameters(), 'clip_val': 1.0},
                    'fusion': {'params': self.fusion_model.parameters(), 'clip_val': 1.0}
                }

            normalizer = ModalityAwareNormalizer(modalities)

            for batch_idx, (batch1, batch2) in enumerate(zip(loader, trans_train_loader)):
                inputs = {k: v.to(self.device) for k, v in batch1.items()}
                seq_x = batch2["sequence"].to(self.device)
                stat_x = batch2["flow_feature"].to(self.device)
                labels = batch2["label"].to(self.device)

                # Zero gradients
                self.text_optim.zero_grad()
                self.trans_optim.zero_grad()
                self.fusion_optim.zero_grad()

                # Text encoder forward
                cls_feat = self.text_model(
                    inputs["input_ids"], inputs["attention_mask"]
                )

                # Transformer forward
                trans_feat = self.trans_model(stat_x, seq_x)

                # Fusion and loss
                loss, final_output = self.fusion_model(cls_feat, trans_feat, inputs["label"])
                if self.use_parallel:
                    loss = loss.sum()

                loss.backward()

                # Modality-aware gradient normalization
                normalizer.normalize()

                # Gradient clipping
                text_params = self.text_model.module.parameters() if self.use_parallel else self.text_model.parameters()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, text_params), 5.0)
                torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), 1.0)

                # Optimizer step
                self.text_optim.step()
                self.trans_optim.step()
                self.fusion_optim.step()

                self.text_scheduler.step()
                self.trans_scheduler.step()
                self.fusion_scheduler.step()

                total_loss += loss.item()
                all_preds.extend(final_output.argmax(dim=1).cpu().numpy())
                all_labels.extend(inputs["label"].cpu().numpy())

            metrics = self._compute_metrics(total_loss / len(loader), all_preds, all_labels)
            self._log_metrics(metrics, prefix="train")

            return metrics, all_preds, all_labels, self.text_model, self.trans_model, self.fusion_model

    def _compute_metrics(self, avg_loss, preds, labels):
        """
        Compute classification metrics.
        """
        return {
            "loss": avg_loss,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0)
        }

    def _log_metrics(self, metrics: dict, prefix: str = "train"):
        """
        Log metrics to SwanLab with a unified interface.
        """
        log_data = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.swanlab.log(log_data)
