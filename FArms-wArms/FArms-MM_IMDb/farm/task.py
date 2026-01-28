import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import clip
import json
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from typing import Tuple, Optional
import warnings
from sklearn.metrics import f1_score, average_precision_score

warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = 200000000

# Dataset paths
MMIMDB_ROOT = "/home/office/Downloads/_Dataset/mmimdb/"
SPLIT_FILE = os.path.join(MMIMDB_ROOT, "split.json")
DATASET_DIR = os.path.join(MMIMDB_ROOT, "dataset")

# Genre list (will be loaded from data)
GENRE_LIST = None
GENRE_TO_IDX = None


def analyze_dataset():
    """Analyze dataset to get genre list"""
    global GENRE_LIST, GENRE_TO_IDX
    
    if GENRE_LIST is not None:
        return GENRE_LIST, GENRE_TO_IDX
    
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    
    all_genres = set()
    for split in ['train', 'dev', 'test']:
        for sample_id in splits[split]:
            json_path = os.path.join(DATASET_DIR, f"{sample_id}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    genres = metadata.get('genres', [])
                    all_genres.update(genres)
    
    GENRE_LIST = sorted(list(all_genres))
    GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_LIST)}
    
    print(f"\n{'='*80}")
    print(f"DISCOVERED {len(GENRE_LIST)} GENRES FROM DATASET")
    print(f"{'='*80}\n")
    
    return GENRE_LIST, GENRE_TO_IDX


class MMIMDbDataset(Dataset):
    def __init__(self, sample_ids, genre_list, genre_to_idx, 
                 missing_config, img_size=224, seed=42):
        self.sample_ids = sample_ids
        self.genre_list = genre_list
        self.genre_to_idx = genre_to_idx
        self.missing_config = missing_config
        self.img_size = img_size
        self.seed = seed
        
        self.parse_missing_config()
        self.modality_availability = self._precompute_modality_availability()
        
        # CLIP preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def parse_missing_config(self):
        if self.missing_config == '100_image_100_text':
            self.image_ratio = 1.0
            self.text_ratio = 1.0
            self.complex_mode = False
        elif self.missing_config.startswith('100_image_'):
            self.image_ratio = 1.0
            text_pct = int(self.missing_config.split('_')[2])
            self.text_ratio = text_pct / 100.0
            self.complex_mode = False
        elif self.missing_config.endswith('_100_text'):
            img_pct = int(self.missing_config.split('_')[0])
            self.image_ratio = img_pct / 100.0
            self.text_ratio = 1.0
            self.complex_mode = False
        elif self.missing_config.startswith('complex_'):
            parts = self.missing_config.split('_')
            self.alpha = int(parts[1]) / 100.0
            self.beta = int(parts[2]) / 100.0
            self.gamma = int(parts[3]) / 100.0
            self.complex_mode = True
            self.image_ratio = None
            self.text_ratio = None
        else:
            raise ValueError(f"Unknown missing config: {self.missing_config}")
    
    def _precompute_modality_availability(self):
        rng = np.random.RandomState(self.seed)
        availability = {}
        
        if self.complex_mode:
            for idx in range(len(self.sample_ids)):
                rand_val = rng.random()
                if rand_val < self.alpha:
                    availability[idx] = (True, True)
                elif rand_val < self.alpha + self.beta:
                    availability[idx] = (True, False)
                else:
                    availability[idx] = (False, True)
        else:
            for idx in range(len(self.sample_ids)):
                has_image = rng.random() < self.image_ratio
                has_text = rng.random() < self.text_ratio
                availability[idx] = (has_image, has_text)
        
        return availability
    
    def load_sample_data(self, sample_id):
        json_path = os.path.join(DATASET_DIR, f"{sample_id}.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        plot = metadata.get('plot', [''])[0] if metadata.get('plot') else ''
        if isinstance(plot, list):
            plot = ' '.join(plot)
        
        genres = metadata.get('genres', [])
        
        label = torch.zeros(len(self.genre_list))
        for genre in genres:
            if genre in self.genre_to_idx:
                label[self.genre_to_idx[genre]] = 1.0
        
        img_path = os.path.join(DATASET_DIR, f"{sample_id}.jpeg")
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = self.image_transform(image)
        except Exception:
            image = torch.zeros(3, self.img_size, self.img_size)
        
        return image, plot, label
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image, text, label = self.load_sample_data(sample_id)
        
        has_image, has_text = self.modality_availability[idx]
        
        if not has_image:
            image = torch.zeros_like(image)
        if not has_text:
            text = ""
        
        return {
            'image': image,
            'text': text,
            'label': label,
            'has_image': has_image,
            'has_text': has_text
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'texts': [item['text'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch]),
        'has_image': [item['has_image'] for item in batch],
        'has_text': [item['has_text'] for item in batch]
    }


def load_data(partition_id: int, num_partitions: int, missing_config):
    """Load federated partition of MMIMDb dataset"""
    genre_list, genre_to_idx = analyze_dataset()
    
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    
    train_ids = splits['train']
    
    # Partition the training data
    partition_size = len(train_ids) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(train_ids)
    partition_train_ids = train_ids[start_idx:end_idx]
    
    # Use 20% for validation
    val_size = int(0.2 * len(partition_train_ids))
    val_ids = partition_train_ids[:val_size]
    train_ids_final = partition_train_ids[val_size:]
    
    train_dataset = MMIMDbDataset(train_ids_final, genre_list, genre_to_idx, missing_config)
    val_dataset = MMIMDbDataset(val_ids, genre_list, genre_to_idx, missing_config)
    
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                            collate_fn=collate_fn, num_workers=2)
    valloader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                          collate_fn=collate_fn, num_workers=2)
    
    return trainloader, valloader


def load_test_data(missing_config):
    """Load complete test set for final evaluation"""
    genre_list, genre_to_idx = analyze_dataset()
    
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    
    test_ids = splits['test']
    
    test_dataset = MMIMDbDataset(test_ids, genre_list, genre_to_idx, missing_config)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)
    
    return testloader


# Model Components
class FrozenCLIPImageEncoder(nn.Module):
    """Frozen CLIP image encoder"""
    def __init__(self, device):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = device
    
    def forward(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
            return features.float()


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder"""
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = device
    
    def forward(self, texts):
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            return features.float()


class FusionModule(nn.Module):
    """Multi-Head Self-Attention fusion"""
    def __init__(self, input_dims, output_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.output_dim = output_dim
        
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, *features):
        projected = []
        for i, feat in enumerate(features):
            proj = self.modality_projections[i](feat)
            projected.append(proj)
        
        modality_sequence = torch.stack(projected, dim=1)
        
        attn_output = modality_sequence
        for attn_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn_layer(attn_output, attn_output, attn_output)
            attn_output = layer_norm(attn_output + attn_out)
        
        fused = attn_output.mean(dim=1)
        output = self.relu(self.output_proj(fused))
        
        return output


class Simulator(nn.Module):
    """Three-layer MLP simulator"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


class Net(nn.Module):
    """Cross-Modal Simulation Model"""
    def __init__(self, num_classes=None, device='cuda'):
        super().__init__()
        
        if num_classes is None:
            genre_list, _ = analyze_dataset()
            num_classes = len(genre_list)
        
        self.image_encoder = FrozenCLIPImageEncoder(device)
        self.text_encoder = FrozenCLIPTextEncoder(device)
        
        self.image_dim = 512
        self.text_dim = 512
        self.fusion_dim = 256
        
        self.fuse_it = FusionModule([self.image_dim, self.text_dim], 
                                     output_dim=self.fusion_dim, 
                                     num_heads=4, num_layers=2)
        
        self.sim_t_i = Simulator(self.text_dim, self.image_dim, hidden_dim=256)
        self.sim_i_t = Simulator(self.image_dim, self.text_dim, hidden_dim=256)
        
        self.final_fusion = FusionModule([self.image_dim, self.text_dim], 
                                         output_dim=self.fusion_dim,
                                         num_heads=4, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self.num_classes = num_classes
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n{'='*80}")
        print("Cross-Modal Simulation Model")
        print(f"{'='*80}")
        print(f"  CLIP parameters (frozen):     {total_params - trainable_params:,}")
        print(f"  Trainable parameters:         {trainable_params:,}")
        print(f"  Total parameters:             {total_params:,}")
        print(f"{'='*80}\n")
    
    def forward(self, images, texts, has_image, has_text):
        batch_size = images.shape[0]
        device = next(self.parameters()).device
        
        image_feat = torch.zeros(batch_size, self.image_dim).to(device)
        text_feat = torch.zeros(batch_size, self.text_dim).to(device)
        
        image_available_indices = [i for i in range(batch_size) if has_image[i]]
        text_available_indices = [i for i in range(batch_size) if has_text[i]]
        
        if image_available_indices:
            image_batch = images[image_available_indices]
            image_feat[image_available_indices] = self.image_encoder(image_batch)
        
        if text_available_indices:
            text_batch = [texts[i] for i in text_available_indices]
            text_feat[text_available_indices] = self.text_encoder(text_batch)
        
        sim_losses = []
        final_image = image_feat.clone()
        final_text = text_feat.clone()
        
        for i in range(batch_size):
            i_avail = has_image[i]
            t_avail = has_text[i]
            
            if i_avail and t_avail:
                sim_losses.append(F.mse_loss(
                    self.sim_t_i(text_feat[i:i+1]), 
                    image_feat[i:i+1]
                ))
                sim_losses.append(F.mse_loss(
                    self.sim_i_t(image_feat[i:i+1]), 
                    text_feat[i:i+1]
                ))
            elif i_avail and not t_avail:
                final_text[i:i+1] = self.sim_i_t(image_feat[i:i+1])
            elif t_avail and not i_avail:
                final_image[i:i+1] = self.sim_t_i(text_feat[i:i+1])
        
        fused = self.final_fusion(final_image, final_text)
        logits = self.classifier(fused)
        
        avg_sim_loss = torch.mean(torch.stack(sim_losses)) if sim_losses else torch.tensor(0.0).to(device)
        
        return logits, avg_sim_loss


def train(net, trainloader, epochs, lr, alpha, beta, device):
    """Training function"""
    net = net.to(device)
    net.train()
    net.image_encoder.eval()
    net.text_encoder.eval()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), 
        lr=lr
    )
    
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch['images'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            has_image = batch['has_image']
            has_text = batch['has_text']
            
            optimizer.zero_grad()
            
            logits, sim_loss = net(images, texts, has_image, has_text)
            cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            loss = alpha * sim_loss + beta * cls_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device):
    """Evaluation function"""
    net = net.to(device)
    net.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in testloader:
            images = batch['images'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            has_image = batch['has_image']
            has_text = batch['has_text']
            
            logits, _ = net(images, texts, has_image, has_text)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    f1_micro = f1_score(all_labels.flatten(), all_preds.flatten(), average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return f1_micro, f1_macro


def test_final(net, testloader, device):
    """Final test evaluation with detailed metrics"""
    net = net.to(device)
    net.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\n{'#'*80}")
    print(f"# FINAL TEST SET EVALUATION")
    print(f"{'#'*80}\n")
    
    with torch.no_grad():
        for batch in testloader:
            images = batch['images'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device)
            has_image = batch['has_image']
            has_text = batch['has_text']
            
            logits, _ = net(images, texts, has_image, has_text)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    f1_micro = f1_score(all_labels.flatten(), all_preds.flatten(), average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    map_score = average_precision_score(all_labels, all_probs, average='macro')
    
    print(f"\n{'='*80}")
    print(f"FINAL TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"\nCore Metrics:")
    print(f"  F1-Micro:          {f1_micro:.4f}")
    print(f"  F1-Macro:          {f1_macro:.4f}")
    print(f"  mAP:               {map_score:.4f}")
    print(f"{'='*80}\n")
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'map': map_score
    }
