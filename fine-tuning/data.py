import os.path

from tqdm import tqdm
from transformers import BertTokenizer
from typing import List, Tuple, Dict
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sklearn.utils import resample
from collections import defaultdict


class NetworkFlowDataset(Dataset):
    """Dataset class for processing network flow payload text"""

    def __init__(self, data: List[Tuple], tokenizer: BertTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        texts, label = self.data[idx]
        concatenated_text = " ".join(texts)

        # Encode concatenated payload text
        encoding = self.tokenizer(
            concatenated_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # shape: [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class TransformerPacketDataset(Dataset):
    """Dataset for packet-level temporal features used by Transformer"""

    def __init__(self, data):
        self.X = torch.tensor(data["sequences"], dtype=torch.float32)
        self.y = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "sequence": self.X[idx],  # shape: [seq_len, feature_dim]
            "label": self.y[idx]
        }


class CombinedFlowDataset(Dataset):
    """Combined dataset for packet sequences and flow-level statistical features"""

    def __init__(self, flow_features, packet_sequences, labels):
        assert len(packet_sequences) == len(flow_features) == len(labels), \
            "Inconsistent data lengths"
        self.packet_sequences = torch.tensor(packet_sequences, dtype=torch.float32)
        self.flow_features = torch.tensor(flow_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        packet_seq = self.packet_sequences[idx]          # shape: [seq_len, feat_dim]
        flow_stat = self.flow_features[idx].unsqueeze(0) # shape: [1, stat_feat_dim]
        label = self.labels[idx]
        return {
            "sequence": packet_seq,      # input for DualTransformer
            "flow_feature": flow_stat,   # input for FlowTransformer
            "label": label
        }


# ===================== BERT data loading =====================

def load_label_dict(data_path: str) -> Dict:
    """Load or build label-to-id mapping"""
    if os.path.exists(f"{data_path}/splitcap/label_trans_dict.json"):
        with open(f"{data_path}/splitcap/label_trans_dict.json") as f:
            label_dict = json.load(f)
    else:
        label_set = set()
        with open(f"{data_path}/splitcap/tcn_payload.jsonl", 'r') as f:
            for line in tqdm(f, desc="Reading labels"):
                data = json.loads(line)
                if data['label'] != ".ipynb_checkpoints":
                    label_set.add(data['label'])

        # Build label → id mapping
        label_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        with open(f"{data_path}/splitcap/label_dict.json", "w") as file:
            json.dump(label_dict, file)

    return label_dict


def load_flow_data(data_path: str, max_samples: int = 200) -> Tuple:
    """Load network flow payload data"""
    label_dict = load_label_dict(data_path)
    samples = []

    with open(f"{data_path}/splitcap/tcn_payload.jsonl", 'r') as f2:
        for line in tqdm(f2, desc="Loading flow data"):
            data = json.loads(line)
            if data["label"] == ".ipynb_checkpoints":
                continue
            packets = data["payloads"]
            samples.append((packets, label_dict[data["label"]]))

    return samples, list(label_dict.keys()), label_dict


# ===================== Transformer data augmentation =====================

def add_noise(X_batch, noise_level=0.05):
    """Add Gaussian noise to packet features"""
    if isinstance(X_batch, np.ndarray):
        X_batch = torch.tensor(X_batch, dtype=torch.float32)
    noise = torch.randn_like(X_batch, device=X_batch.device) * noise_level
    return X_batch + noise


def mask_packets(X_batch, mask_prob=0.1):
    """Randomly mask packets (packet drop simulation)"""
    batch_size, seq_len, feature_len = X_batch.shape
    for i in range(batch_size):
        num_masked = int(seq_len * mask_prob)
        masked_indices = np.random.choice(seq_len, num_masked, replace=False)
        X_batch[i, masked_indices] = 0
    return X_batch


def shuffle_packets(X_batch, shuffle_prob=0.1):
    """Randomly shuffle packet order"""
    batch_size, seq_len, feature_len = X_batch.shape
    for i in range(batch_size):
        if torch.rand(1).item() < shuffle_prob:
            indices = torch.randperm(seq_len)
            X_batch[i] = X_batch[i][indices]
    return X_batch


def augment_data(X_batch, noise_level=0.05, mask_prob=0.1, shuffle_prob=0.1):
    """Apply noise, masking, and shuffling augmentation"""
    X_batch = add_noise(X_batch, noise_level)
    X_batch = mask_packets(X_batch, mask_prob)
    X_batch = shuffle_packets(X_batch, shuffle_prob)
    return X_batch


def get_balanced_trans_dataset(
        trans_dataset,
        target_class_count=1000,
        noise_level=0.05,
        mask_prob=0.1,
        shuffle_prob=0.1):
    """
    Balance TransformerPacketDataset by over/under-sampling and augmentation
    """
    X = trans_dataset.X.numpy()
    y = trans_dataset.y.numpy()

    balanced_X = []
    balanced_y = []

    for cls in np.unique(y):
        cls_X = X[y == cls]
        cls_y = y[y == cls]

        if len(cls_X) < target_class_count:
            cls_X_resampled, cls_y_resampled = resample(
                cls_X, cls_y, replace=True,
                n_samples=target_class_count, random_state=42)
        else:
            cls_X_resampled = cls_X[:target_class_count]
            cls_y_resampled = cls_y[:target_class_count]

        cls_X_resampled = augment_data(
            cls_X_resampled, noise_level, mask_prob, shuffle_prob)

        balanced_X.append(cls_X_resampled)
        balanced_y.append(cls_y_resampled)

    balanced_data = {
        "sequences": np.concatenate(balanced_X, axis=0),
        "labels": np.concatenate(balanced_y, axis=0)
    }

    print(f"Balanced all classes to {target_class_count} samples")
    return TransformerPacketDataset(balanced_data)


# ===================== Feature-level augmentation =====================

def augment_sample(sample, noise_level=0.01):
    """Apply Gaussian noise to a single feature vector"""
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32)
    noise = torch.randn_like(sample, device=sample.device) * noise_level
    return sample + noise


# ===================== Text masking augmentation =====================

def random_token_masking(
        texts: List[str],
        mask_prob=0.15,
        mask_token="[MASK]") -> List[str]:
    """
    Apply token-level random masking to payload text
    Assumes payloads are whitespace-tokenized
    """
    masked = []
    for payload in texts:
        tokens = payload.split()
        new_tokens = [
            mask_token if random.random() < mask_prob else t
            for t in tokens
        ]
        masked.append(" ".join(new_tokens))
    return masked


def get_balanced_masked_bert_dataset(
        samples: List[Tuple[List[str], int]],
        tokenizer: BertTokenizer,
        target_count: int,
        mask_prob: float = 0.15,
        max_length: int = 512):
    """
    Balance BERT payload samples by label.
    Oversampled samples are augmented via random token masking.
    """
    label_to_samples = defaultdict(list)
    for s in samples:
        label_to_samples[s[1]].append(s)

    new_samples = []

    for label, s_list in label_to_samples.items():
        if len(s_list) >= target_count:
            selected = s_list[:target_count]
        else:
            selected = []
            needed = target_count - len(s_list)
            repeats = resample(
                s_list, n_samples=needed,
                replace=True, random_state=42)
            for orig_texts, _ in repeats:
                masked_texts = random_token_masking(orig_texts, mask_prob)
                selected.append((masked_texts, label))
            selected += s_list

        new_samples.extend(selected)

    print(f"Resampled each class to {target_count} samples")
    return new_samples
    # return NetworkFlowDataset(new_samples, tokenizer, max_length)
