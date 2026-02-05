import os
import torch
from torch.utils.data import DataLoader, random_split
from data import (
    NetworkFlowDataset,
    load_flow_data,
    TransformerPacketDataset,
    get_balanced_masked_bert_dataset
)
from data import balance_and_augment_data, augment_and_balance, CombinedFlowDataset
from models import TextFeatureExtractor, FusionModel, UnifiedFlowModel
from trainer import FlowTrainer
from transformers import BertTokenizer
import numpy as np
import swanlab
from transformers import get_scheduler
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse
import torch_optimizer as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# =========================
# Configuration
# =========================
config = {
    "bert_path": "/root/netgpt/model",
    "data_path": "/root/autodl-tmp/data/USTC-TFC2016-master/Malware/",
    "tokenizer_path": "/root/netgpt/vocab/bert_tokenizer",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 32,
    "text_lr": 3e-5,
    "trans_lr": 8e-5,
    "fusion_lr": 5e-6,
    "epochs": 30,
    "feature_num": 28,
    "target_count": 500,        # Target number of samples per class
    "trans_max_epoch": 30,
    "text_max_epoch": 10,
    "fusion_max_epoch": 20,
    "trans_patience": 10,
    "trans_hidden_size": 256,
    "fusion_hidden_size": 256,
    "bert_use_multiclassifier": False,
    "use_parallel": False,
    "start_freeze_layer": 8,
    "use_unfreeze": False,
    "continue_train": False
}


def save_model(model, optimizer, scheduler, epoch, model_name="model.pth", use_parallel=False):
    """Save model, optimizer, and scheduler states"""
    if use_parallel:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.module.state_dict(),
            'scheduler_state_dict': scheduler.module.state_dict(),
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
    torch.save(checkpoint, model_name)


def load_model(model, optimizer, scheduler, device, model_name="model.pth", use_parallel=False):
    """Load model, optimizer, and scheduler states"""
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Move optimizer states to the target device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    start_epoch = checkpoint['epoch']
    print(f"Model loaded from {model_name}")
    return start_epoch, model, optimizer, scheduler


def main():
    start_epoch = 0
    device = config["device"]

    # =========================
    # Load pre-extracted features
    # =========================
    data = np.load(config["data_path"] + "/splitcap/all_feature.npz", allow_pickle=True)
    packet_sequences = data["sequences"]
    flow_labels = data["labels"]
    flow_features = data["stat_features"]

    # =========================
    # Prepare Transformer datasets (statistical + packet-level features)
    # =========================
    X1_bal, y_bal = augment_and_balance(
        flow_features,
        flow_labels,
        target_count=config["target_count"],
        noise_level=0.01
    )

    X2_bal, y_bal = balance_and_augment_data(
        packet_sequences,
        flow_labels,
        config["target_count"],
        noise_level=0.05,
        mask_prob=0.1,
        shuffle_prob=0.1
    )

    # Split into training and validation sets
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_bal, X2_bal, y_bal,
        test_size=0.2,
        stratify=y_bal,
        random_state=3407
    )

    trans_train_set = CombinedFlowDataset(X1_train, X2_train, y_train)
    trans_val_set = CombinedFlowDataset(X1_val, X2_val, y_val)

    trans_train_loader = DataLoader(
        trans_train_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True
    )
    trans_val_loader = DataLoader(
        trans_val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True
    )

    # =========================
    # Load BERT text data
    # =========================
    samples, labels, label_dict = load_flow_data(config["data_path"])
    bert_y = [y for x, y in samples]

    print("Label distribution (Transformer):", np.unique(flow_labels, return_counts=True))
    print("Label distribution (BERT):", np.unique(bert_y, return_counts=True))

    # Consistency check between modalities
    assert (flow_labels == bert_y).all(), "Label mismatch between modalities"

    num_classes = len(labels)

    # Use a custom classifier head if the number of classes is large
    if num_classes > 20:
        config["bert_use_multiclassifier"] = True

    print(f"Total number of classes: {num_classes}")

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer_path"])

    new_samples = get_balanced_masked_bert_dataset(
        samples,
        tokenizer=tokenizer,
        target_count=config["target_count"],
        mask_prob=0.2
    )

    bert_train, bert_val = train_test_split(
        new_samples,
        test_size=0.2,
        stratify=y_bal,
        random_state=3407
    )

    train_set = NetworkFlowDataset(bert_train, tokenizer)
    val_set = NetworkFlowDataset(bert_val, tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True
    )

    # =========================
    # Model initialization
    # =========================
    text_model = TextFeatureExtractor(
        config["bert_path"],
        num_classes,
        config["bert_use_multiclassifier"],
        dropout_prob=0.5,
        start_freeze_layer=config["start_freeze_layer"]
    )

    trans_model = UnifiedFlowModel(
        stat_feat_dim=42,
        seq_feat_dim=28,
        seq_len=100,
        hidden_dim=config["trans_hidden_size"],
        num_classes=num_classes
    )

    fusion_model = FusionModel(
        hidden_size=config["fusion_hidden_size"],
        trans_hidden_size=config["trans_hidden_size"],
        num_classes=num_classes,
        dropout=0.3
    )

    # =========================
    # Optimizers
    # =========================
    optimizer_grouped_parameters = [
        {"params": text_model.bert.bert.encoder.layer[:8].parameters(), "lr": config["text_lr"] / 5},
        {"params": text_model.bert.bert.encoder.layer[8:].parameters(), "lr": config["text_lr"]},
        {"params": text_model.bert.classifier.parameters(), "lr": config["text_lr"] * 2},
    ]

    text_optim = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    trans_optim = torch.optim.AdamW(
        trans_model.parameters(),
        lr=config["trans_lr"],
        weight_decay=0.01
    )

    fusion_optim = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=config["fusion_lr"]
    )

    # =========================
    # Learning rate schedulers
    # =========================
    num_training_steps = config["epochs"] * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    text_scheduler = get_scheduler(
        "cosine",
        optimizer=text_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    trans_scheduler = get_scheduler(
        "cosine",
        optimizer=trans_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    fusion_scheduler = get_scheduler(
        "cosine",
        optimizer=fusion_optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # =========================
    # Multi-GPU support
    # =========================
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        config["use_parallel"] = True
        text_model = nn.DataParallel(text_model)
        trans_model = nn.DataParallel(trans_model)
        fusion_model = nn.DataParallel(fusion_model)

    text_model.to(device)
    trans_model.to(device)
    fusion_model.to(device)

    # =========================
    # Initialize SwanLab experiment
    # =========================
    swanlab.init(
        project="bert-triformer",
        experiment_name=config["data_path"].split("/")[-3] + "-" + config["data_path"].split("/")[-2],
        config={
            "model_type": "fusion_model",
            "device": str(device)
        }
    )

    # =========================
    # Training
    # =========================
    trainer = FlowTrainer(
        num_classes,
        text_model,
        trans_model,
        fusion_model,
        text_optim,
        trans_optim,
        fusion_optim,
        text_scheduler,
        trans_scheduler,
        fusion_scheduler,
        device,
        config["use_parallel"],
        config["start_freeze_layer"],
        swanlab
    )

    trainer.warmup_trans_model(
        trans_train_loader,
        max_epochs=config["trans_max_epoch"],
        val_loader=trans_val_loader,
        patience=config["trans_patience"]
    )

    trainer.warmup_fusion_model(
        train_loader,
        trans_train_loader,
        max_epochs=config["fusion_max_epoch"]
    )

    # Epoch-level progress bar
    epoch_pbar = tqdm(total=config["epochs"], desc="Training Progress", initial=start_epoch)

    for epoch in range(start_epoch, start_epoch + config["epochs"]):
        train_metrics, _, _, text_model, trans_model, fusion_model = trainer.train_epoch(
            train_loader,
            trans_train_loader
        )

        val_metrics, preds, labels = trainer.evaluate(val_loader, trans_val_loader)

        epoch_pbar.update(1)
        epoch_pbar.set_postfix({
            'Epoch': f'{epoch + 1}/{config["epochs"]}',
            'Train Loss': f'{train_metrics["loss"]:.4f}',
            'Val Loss': f'{val_metrics["loss"]:.4f}',
            'Val Acc': f'{val_metrics["accuracy"]:.4f}',
            'Val F1': f'{val_metrics["f1"]:.4f}'
        })

        # Save checkpoints periodically
        if (epoch + 1) % 5 == 0 or epoch == start_epoch + config["epochs"] - 1:
            save_model(
                text_model,
                text_optim,
                text_scheduler,
                epoch,
                model_name=config["data_path"] + "/splitcap/text_model.pth",
                use_parallel=config["use_parallel"]
            )

    epoch_pbar.close()

    # =========================
    # Confusion matrix
    # =========================
    cm = confusion_matrix(labels, preds)
    print(cm)
    print("Training completed!")

    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset selection flags
    parser.add_argument("--ServiceVPN", action="store_true", help="Use Service-VPN datasets")
    parser.add_argument("--ServiceNonVPN", action="store_true", help="Use Service-NonVPN datasets")
    parser.add_argument("--AppVPN", action="store_true", help="Use App-VPN datasets")
    parser.add_argument("--AppNonVPN", action="store_true", help="Use App-NonVPN datasets")
    parser.add_argument("--Tor", action="store_true", help="Use Tor datasets")
    parser.add_argument("--NonTor", action="store_true", help="Use NonTor datasets")
    parser.add_argument("--Benign", action="store_true", help="Use Benign datasets")
    parser.add_argument("--Malware", action="store_true", help="Use Malware datasets")
    parser.add_argument("--Flood", action="store_true", help="Use Flood datasets")
    parser.add_argument("--RTSPBruteForce", action="store_true", help="Use RTSP-Brute-Force datasets")
    parser.add_argument("--datacon2020", action="store_true", help="Use datacon2020 datasets")
    parser.add_argument("--datacon2021part1", action="store_true", help="Use datacon2021 part1 datasets")
    parser.add_argument("--datacon2021part2", action="store_true", help="Use datacon2021 part2 datasets")
    parser.add_argument("--CrossPlatformandroid", action="store_true", help="Use CrossPlatform Android datasets")
    parser.add_argument("--CrossPlatformios", action="store_true", help="Use CrossPlatform iOS datasets")
    parser.add_argument("--NUDT", action="store_true", help="Use NUDT datasets")

    args = parser.parse_args()

    if args.Malware:
        config["data_path"] = "/root/autodl-tmp/berttrans/USTC-TFC2016/Malware/"

    main()
