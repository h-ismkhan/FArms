import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from torchvision.models import resnet18
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

# FIX 1: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# FIX 2: Suppress warnings
import warnings
warnings.filterwarnings('ignore', message='In 2.9, this function')

# Import the MOSI dataset for regression
from mosi_reg import MOSIDatasetRegression

dsspath = "/home/"

# ============================================================================
# FROZEN ENCODERS
# ============================================================================
class FrozenTextEncoder(nn.Module):
    """Frozen BERT encoder for text"""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, texts):
        with torch.no_grad():
            encoding = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            )
            encoding = {k: v.to(next(self.model.parameters()).device) for k, v in encoding.items()}
            outputs = self.model(**encoding)
            return outputs.last_hidden_state[:, 0, :]

from transformers import Wav2Vec2FeatureExtractor, WavLMModel

class FrozenAudioEncoder(nn.Module):
    """Frozen WavLM-Large encoder for audio"""
    def __init__(self):
        super().__init__()
        # Use FeatureExtractor instead of Processor (no tokenizer for audio-only models)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, audio_batch):
        with torch.no_grad():
            processed = []
            for audio in audio_batch:
                if isinstance(audio, torch.Tensor):
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)  # To mono
                    waveform = audio.squeeze().cpu().numpy()
                else:
                    waveform = audio
                
                if waveform.ndim > 1:
                    waveform = waveform.flatten()
                
                # Use feature_extractor instead of processor
                inputs = self.feature_extractor(
                    waveform, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=False
                )
                processed.append(inputs.input_values.squeeze(0))
            
            # Pad to max length in batch
            max_len = max(p.shape[-1] for p in processed)
            padded = torch.stack([
                F.pad(p, (0, max_len - p.shape[-1])) for p in processed
            ]).to(next(self.model.parameters()).device)
            
            outputs = self.model(padded)
            return outputs.last_hidden_state.mean(dim=1)  # Mean pooling (shape: (batch_size, 1024))
        
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class FrozenVideoEncoder(nn.Module):
    """Frozen CLIP Vision encoder for video frames - extract features from sampled frames"""
    def __init__(self, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        
        # Use CLIP Vision model instead of X-CLIP for simpler processing
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def forward(self, video_batch):
        device = next(self.model.parameters()).device
        batch_size = len(video_batch)
        
        with torch.no_grad():
            all_features = []
            
            for idx, frames in enumerate(video_batch):
                if len(frames) == 0:
                    # Create zero features for empty videos
                    # CLIP Vision outputs 768-dim features
                    frame_features = torch.zeros(self.num_frames, 768)
                else:
                    # Sample exactly self.num_frames frames
                    if len(frames) >= self.num_frames:
                        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
                    else:
                        repeat_factor = (self.num_frames + len(frames) - 1) // len(frames)
                        indices = np.tile(np.arange(len(frames)), repeat_factor)[:self.num_frames]
                    
                    # Convert frames to PIL Images and process
                    frame_features_list = []
                    for i in indices:
                        frame = frames[i][:, :, ::-1]  # BGR → RGB
                        pil_image = Image.fromarray(frame.astype('uint8'))
                        
                        # Process single frame
                        inputs = self.processor(images=pil_image, return_tensors="pt")
                        pixel_values = inputs["pixel_values"].to(device)
                        
                        # Extract features for this frame
                        outputs = self.model(pixel_values=pixel_values)
                        frame_feat = outputs.pooler_output  # [1, 768]
                        frame_features_list.append(frame_feat.squeeze(0))
                    
                    # Stack frame features: [num_frames, 768]
                    frame_features = torch.stack(frame_features_list)
                
                # Average pool across frames to get video-level feature: [768]
                video_feature = frame_features.mean(dim=0)
                all_features.append(video_feature)
            
            # Stack all videos: [batch_size, 768]
            video_feats = torch.stack(all_features).to(device)
            
            # Zero out features for empty videos
            for idx, frames in enumerate(video_batch):
                if len(frames) == 0:
                    video_feats[idx] = 0.0
            
            return video_feats

# ============================================================================
# TRAINABLE COMPONENTS
# ============================================================================

class FusionModule(nn.Module):
    """Multi-Head Self-Attention fusion"""
    def __init__(self, input_dims, output_dim=256, num_heads=4, num_layers=2):
        super().__init__()
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
        projected = [self.modality_projections[i](feat) for i, feat in enumerate(features)]
        modality_sequence = torch.stack(projected, dim=1)
        
        attn_output = modality_sequence
        for attn_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attn_layer(attn_output, attn_output, attn_output)
            attn_output = layer_norm(attn_output + attn_out)
        
        fused = attn_output.mean(dim=1)
        return self.relu(self.output_proj(fused))


class Simulator(nn.Module):
    """Three-layer MLP simulator"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()      
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout_dpt1 = nn.Dropout(p=0.1)
        self.dropout_dpt01 = nn.Dropout(p=0.01)  
    
    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))        
        #return self.fc3(self.dropout_dpt1(self.relu2(self.fc2(self.dropout_dpt1(self.relu1(self.fc1(x)))))))
        #return self.fc3(self.dropout_dpt1(self.relu2(self.fc2(self.relu1(self.fc1(x))))))
        


# ============================================================================
# MAIN MODEL
# ============================================================================

class Net(nn.Module):
    """Complete model with cross-modal simulation for regression"""
    def __init__(self):
        super().__init__()
        
        # Frozen encoders
        self.text_encoder = FrozenTextEncoder()
        self.audio_encoder = FrozenAudioEncoder()
        self.video_encoder = FrozenVideoEncoder()
        
        # Feature dimensions
        self.text_dim = 768
        self.audio_dim = 1024
        self.video_dim = 768
        
        # Fusion modules for simulation (Type 1)
        self.fuse_ta = FusionModule([self.text_dim, self.audio_dim])
        self.fuse_tv = FusionModule([self.text_dim, self.video_dim])
        self.fuse_av = FusionModule([self.audio_dim, self.video_dim])
        
        # Simulators
        self.sim_a_t = Simulator(self.audio_dim, self.text_dim)
        self.sim_v_t = Simulator(self.video_dim, self.text_dim)
        self.sim_av_t = Simulator(256, self.text_dim)
        
        self.sim_t_a = Simulator(self.text_dim, self.audio_dim)
        self.sim_v_a = Simulator(self.video_dim, self.audio_dim)
        self.sim_tv_a = Simulator(256, self.audio_dim)
        
        self.sim_t_v = Simulator(self.text_dim, self.video_dim)
        self.sim_a_v = Simulator(self.audio_dim, self.video_dim)
        self.sim_ta_v = Simulator(256, self.video_dim)
        
        # Final fusion and regressor (Type 2)
        self.final_fusion = FusionModule([self.text_dim, self.audio_dim, self.video_dim])
        self.regressor = nn.Sequential(
            nn.Linear(256, 1),
            nn.Hardtanh(min_val=-3.0, max_val=3.0)
        )
        
        # Count parameters by component
        sim_params = sum(p.numel() for p in self.get_simulation_parameters())
        reg_params = sum(p.numel() for p in self.get_regression_parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        print(f"\n{'='*80}")
        print("CROSS-MODAL SIMULATION MODEL - REGRESSION (SEPARATE OPTIMIZERS)")
        print(f"{'='*80}")
        print(f"  Frozen encoder parameters:        {frozen_params:,}")
        print(f"  Simulation parameters:            {sim_params:,}")
        print(f"  Regression parameters:            {reg_params:,}")
        print(f"  Total trainable parameters:       {sim_params + reg_params:,}")
        print(f"  Total parameters:                 {frozen_params + sim_params + reg_params:,}")
        print(f"{'='*80}\n")
    
    def get_simulation_parameters(self):
        """Get parameters for simulation components"""
        params = []
        # Simulation fusion modules
        params.extend(self.fuse_ta.parameters())
        params.extend(self.fuse_tv.parameters())
        params.extend(self.fuse_av.parameters())
        # All simulators
        params.extend(self.sim_a_t.parameters())
        params.extend(self.sim_v_t.parameters())
        params.extend(self.sim_av_t.parameters())
        params.extend(self.sim_t_a.parameters())
        params.extend(self.sim_v_a.parameters())
        params.extend(self.sim_tv_a.parameters())
        params.extend(self.sim_t_v.parameters())
        params.extend(self.sim_a_v.parameters())
        params.extend(self.sim_ta_v.parameters())
        return params
    
    def get_text_simulation_parameters(self):
        """Get parameters for simulation components"""
        params = []
        # Simulation fusion modules
        params.extend(self.fuse_av.parameters())
        params.extend(self.fuse_ta.parameters())
        params.extend(self.fuse_tv.parameters())
        # All simulators
        params.extend(self.sim_a_t.parameters())
        params.extend(self.sim_v_t.parameters())
        params.extend(self.sim_av_t.parameters())
        return params
    def get_audio_simulation_parameters(self):
        """Get parameters for simulation components"""
        params = []
        # Simulation fusion modules
        params.extend(self.fuse_tv.parameters())
        # All simulators
        params.extend(self.sim_t_a.parameters())
        params.extend(self.sim_v_a.parameters())
        params.extend(self.sim_tv_a.parameters())
        return params
    def get_video_simulation_parameters(self):
        """Get parameters for simulation components"""
        params = []
        # Simulation fusion modules
        params.extend(self.fuse_ta.parameters())
        # All simulators
        params.extend(self.sim_t_v.parameters())
        params.extend(self.sim_a_v.parameters())
        params.extend(self.sim_ta_v.parameters())
        return params
    
    def get_regression_parameters(self):
        """Get parameters for regression components"""
        params = []
        # Final fusion and regressor
        params.extend(self.final_fusion.parameters())
        params.extend(self.regressor.parameters())
        return params
    
    def forward(self, text, audio, video, has_text, has_audio, has_video, return_features=False, skip_verification=False):
        batch_size = len(text)
        device = next(self.parameters()).device
        
        # Extract features
        text_feat = torch.zeros(batch_size, self.text_dim).to(device)
        audio_feat = torch.zeros(batch_size, self.audio_dim).to(device)
        video_feat = torch.zeros(batch_size, self.video_dim).to(device)
        
        text_available_indices = [i for i in range(batch_size) if has_text[i]]
        audio_available_indices = [i for i in range(batch_size) if has_audio[i]]
        video_available_indices = [i for i in range(batch_size) if has_video[i]]
        
        if text_available_indices:
            text_batch = [text[i] for i in text_available_indices]
            text_feat[text_available_indices] = self.text_encoder(text_batch)
        
        if audio_available_indices:
            audio_batch = [audio[i] for i in audio_available_indices]
            audio_feat[audio_available_indices] = self.audio_encoder(audio_batch)
        
        if video_available_indices:
            video_batch = [video[i] for i in video_available_indices]
            video_feat[video_available_indices] = self.video_encoder(video_batch)
        
        # Simulation
        sim_losses_v = []
        sim_losses_t = []
        sim_losses_a = []
        final_text = text_feat.clone()
        final_audio = audio_feat.clone()
        final_video = video_feat.clone()
        
        for i in range(batch_size):
            t_avail, a_avail, v_avail = has_text[i], has_audio[i], has_video[i]
            
            if t_avail and a_avail and v_avail:
                # All three available

                sim_v_from_t = self.sim_t_v(text_feat[i:i+1])#80.2
                fused_tv = self.fuse_tv(text_feat[i:i+1], sim_v_from_t)#80.2
                sim_a_tv = self.sim_tv_a(fused_tv)
                sim_losses_a.append(F.mse_loss(sim_a_tv, audio_feat[i:i+1]))

                sim_t_from_a = self.sim_a_t(audio_feat[i:i+1])#59.025
                fused_ta = self.fuse_ta(sim_t_from_a, audio_feat[i:i+1])#59.025
                sim_v_from_ta = self.sim_ta_v(fused_ta)#59.025
                sim_losses_v.append(F.mse_loss(sim_v_from_ta, video_feat[i:i+1]))

                # sim_t_from_v = self.sim_v_t(video_feat[i:i+1])#ok 4-th run: 0.6778, 9-th run: 0.6816
                # fused_tv = self.fuse_tv(sim_t_from_v, video_feat[i:i+1])#ok 4-th run: 0.6778, 9-th run: 0.6816
                # sim_a_tv = self.sim_tv_a(fused_tv)
                # sim_losses_a.append(F.mse_loss(sim_a_tv, audio_feat[i:i+1]))
                
                sim_a_from_v = self.sim_v_a(video_feat[i:i+1])#3-th run: 0.6970 - 5-th run: 0.6932
                fused_av = self.fuse_av(sim_a_from_v, video_feat[i:i+1])#3-th run: 0.6970 - 5-th run: 0.6932
                sim_t_from_av = self.sim_av_t(fused_av)#3-th run: 0.6970 - 5-th run: 0.6932
                sim_losses_t.append(F.mse_loss(sim_t_from_av, text_feat[i:i+1]))

                fused_av = self.fuse_av(audio_feat[i:i+1], video_feat[i:i+1])
                sim_t = self.sim_av_t(fused_av)
                sim_losses_t.append(F.mse_loss(sim_t, text_feat[i:i+1]))
                
                fused_tv = self.fuse_tv(text_feat[i:i+1], video_feat[i:i+1])
                sim_a = self.sim_tv_a(fused_tv)
                sim_losses_a.append(F.mse_loss(sim_a, audio_feat[i:i+1]))
                
                fused_ta = self.fuse_ta(text_feat[i:i+1], audio_feat[i:i+1])
                sim_v = self.sim_ta_v(fused_ta)
                sim_losses_v.append(F.mse_loss(sim_v, video_feat[i:i+1]))
                
                # we can not add non of these ones, because we already have done
                # e.g., sim_v_from_t indirectly added at the first part: sim_v_from_t->fused_tv->sim_a_tv=>sim_losses_a
                sim_losses_v.append(F.mse_loss(self.sim_t_v(text_feat[i:i+1]), video_feat[i:i+1]))
                sim_losses_t.append(F.mse_loss(self.sim_v_t(video_feat[i:i+1]), text_feat[i:i+1]))
                sim_losses_t.append(F.mse_loss(self.sim_a_t(audio_feat[i:i+1]), text_feat[i:i+1]))
                sim_losses_a.append(F.mse_loss(self.sim_t_a(text_feat[i:i+1]), audio_feat[i:i+1]))
                sim_losses_a.append(F.mse_loss(self.sim_v_a(video_feat[i:i+1]), audio_feat[i:i+1]))
                sim_losses_v.append(F.mse_loss(self.sim_a_v(audio_feat[i:i+1]), video_feat[i:i+1]))
            
            elif t_avail and a_avail and not v_avail:

                sim_v = self.sim_t_v(text_feat[i:i+1])#80.2
                fused_tv = self.fuse_tv(text_feat[i:i+1], sim_v)#80.2
                sim_a_from_tv = self.sim_tv_a(fused_tv)#80.2
                sim_losses_a.append(F.mse_loss(sim_a_from_tv, audio_feat[i:i+1]))
                
                sim_losses_t.append(F.mse_loss(self.sim_a_t(audio_feat[i:i+1]), text_feat[i:i+1]))
                sim_losses_a.append(F.mse_loss(self.sim_t_a(text_feat[i:i+1]), audio_feat[i:i+1]))
                fused_ta = self.fuse_ta(text_feat[i:i+1], audio_feat[i:i+1])
                final_video[i:i+1] = self.sim_ta_v(fused_ta)
            
            elif t_avail and v_avail and not a_avail:           
                
                sim_losses_v.append(F.mse_loss(self.sim_t_v(text_feat[i:i+1]), video_feat[i:i+1]))
                sim_losses_a.append(F.mse_loss(self.sim_v_t(video_feat[i:i+1]), text_feat[i:i+1]))
                fused_tv = self.fuse_tv(text_feat[i:i+1], video_feat[i:i+1])
                final_audio[i:i+1] = self.sim_tv_a(fused_tv)
            
            elif a_avail and v_avail and not t_avail:

                sim_t = self.sim_v_t(video_feat[i:i+1])#ok 4-th run: 0.6778, 9-th run: 0.6816
                fused_tv = self.fuse_tv(sim_t, video_feat[i:i+1])#ok 4-th run: 0.6778, 9-th run: 0.6816                
                sim_a_from_at = self.sim_tv_a(fused_tv)#ok 4-th run: 0.6778, 9-th run: 0.6816
                sim_losses_a.append(F.mse_loss(sim_a_from_at, audio_feat[i:i+1]))

                sim_t = self.sim_a_t(audio_feat[i:i+1])#59.025
                fused_ta = self.fuse_ta(sim_t, audio_feat[i:i+1])#59.025
                sim_v_from_ta = self.sim_ta_v(fused_ta)#59.025
                sim_losses_v.append(F.mse_loss(sim_v_from_ta, video_feat[i:i+1]))

                sim_v = self.sim_t_v(text_feat[i:i+1])#80.2
                fused_tv = self.fuse_tv(text_feat[i:i+1], sim_v)#80.2
                sim_a_from_tv = self.sim_tv_a(fused_tv)#80.2
                sim_losses_a.append(F.mse_loss(sim_a_from_tv, audio_feat[i:i+1]))

                sim_losses_a.append(F.mse_loss(self.sim_v_a(video_feat[i:i+1]), audio_feat[i:i+1]))
                sim_losses_v.append(F.mse_loss(self.sim_a_v(audio_feat[i:i+1]), video_feat[i:i+1]))
                fused_av = self.fuse_av(audio_feat[i:i+1], video_feat[i:i+1])
                final_text[i:i+1] = self.sim_av_t(fused_av)

            elif t_avail and not a_avail and not v_avail:
                sim_v = self.sim_t_v(text_feat[i:i+1])#80.2
                fused_tv = self.fuse_tv(text_feat[i:i+1], sim_v)#80.2
                final_audio[i:i+1] = self.sim_tv_a(fused_tv)#80.2
                final_video[i:i+1] = sim_v#80+-
                # sim_a = self.sim_t_a(text_feat[i:i+1])#80.02
                # fused_ta = self.fuse_ta(text_feat[i:i+1], sim_a)#80.02
                # final_video[i:i+1] = self.sim_ta_v(fused_ta)#80.02
                # final_audio[i:i+1] = sim_a#80.02                
            
            elif a_avail and not t_avail and not v_avail:
                sim_t = self.sim_a_t(audio_feat[i:i+1])#59.025
                fused_ta = self.fuse_ta(sim_t, audio_feat[i:i+1])#59.025
                final_video[i:i+1] = self.sim_ta_v(fused_ta)#59.025
                final_text[i:i+1] = sim_t#59.025
                # sim_v = self.sim_a_v(audio_feat[i:i+1])#56.4
                # fused_av = self.fuse_av(audio_feat[i:i+1], sim_v)#56.4
                # final_text[i:i+1] = self.sim_av_t(fused_av)#56.4
                # final_video[i:i+1] = sim_v#56.4
            
            elif v_avail and not t_avail and not a_avail:
                # sim_t = self.sim_v_t(video_feat[i:i+1])#8-th run: 0.6707
                # fused_tv = self.fuse_tv(sim_t, video_feat[i:i+1])#8-th run: 0.6707
                # final_text[i:i+1] = sim_t#8-th run: 0.6707
                # final_audio[i:i+1] = self.sim_tv_a(fused_tv)#8-th run: 0.6707
                sim_a = self.sim_v_a(video_feat[i:i+1])#3-th run: 0.6970 - 5-th run: 0.6932
                fused_av = self.fuse_av(sim_a, video_feat[i:i+1])#3-th run: 0.6970 - 5-th run: 0.6932
                final_text[i:i+1] = self.sim_av_t(fused_av)#3-th run: 0.6970 - 5-th run: 0.6932
                final_audio[i:i+1] = sim_a#3-th run: 0.6970 - 5-th run: 0.6932
                
        
        fused = self.final_fusion(final_text, final_audio, final_video)
        predictions = self.regressor(fused).squeeze(-1)

        avg_sim_loss_v = torch.mean(torch.stack(sim_losses_v)) if sim_losses_v else torch.tensor(0.0).to(device)
        avg_sim_loss_t = torch.mean(torch.stack(sim_losses_t)) if sim_losses_t else torch.tensor(0.0).to(device)
        avg_sim_loss_a = torch.mean(torch.stack(sim_losses_a)) if sim_losses_a else torch.tensor(0.0).to(device)

        if not skip_verification:
            for i in range(batch_size):
                if torch.all(final_text[i] == 0):
                    raise RuntimeError(f"Sample {i}: Text features are all zeros - not filled!")
                if torch.all(final_audio[i] == 0):
                    raise RuntimeError(f"Sample {i}: Audio features are all zeros - not filled!")
                if torch.all(final_video[i] == 0):
                    raise RuntimeError(f"Sample {i}: Video features are all zeros - not filled!")

        if return_features:
            return predictions, (avg_sim_loss_t + avg_sim_loss_a + avg_sim_loss_v) / 3.0, fused

        return predictions, (avg_sim_loss_t + avg_sim_loss_a + avg_sim_loss_v) / 3.0


def collate_fn(batch):
    """Custom collate function"""
    return {
        'name': [item['name'] for item in batch],
        'text': [item['text'] for item in batch],
        'audio': [item['audio'] for item in batch],
        'video': [item['video'] for item in batch],
        'label': torch.stack([item['label'] for item in batch]),
        'has_text': [item['has_text'] for item in batch],
        'has_audio': [item['has_audio'] for item in batch],
        'has_video': [item['has_video'] for item in batch]
    }


def train_epoch_sim(model, dataloader, lr=5e-4, alpha=1, beta=1.0, device='cuda', info='training'):
    """Train for one epoch with regression loss"""
    model.train()
    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.video_encoder.eval()
    
    optimizer = torch.optim.Adam(
        model.get_simulation_parameters(), 
        lr=lr
    )

    sim_loss_epoch = 0.0
    within_epoch_counter = 0.0
    for batch in tqdm(dataloader, desc=info):
        within_epoch_counter += 1.0
        text = batch['text']
        audio = batch['audio']
        video = batch['video']
        labels = batch['label'].to(device)
        has_text = batch['has_text']
        has_audio = batch['has_audio']
        has_video = batch['has_video']
        
        # Initial forward pass
        _, sim_loss = model(text, audio, video, has_text, has_audio, has_video)
        # Track the original sim_loss
        sim_loss_value = sim_loss.item()
        sim_loss_epoch += sim_loss_value
        
        if sim_loss_value > 0:  # Only if there's actual simulation loss:
            optimizer.zero_grad()
            sim_loss.backward()  # No retain_graph!
            optimizer.step()

    
    return sim_loss_epoch / within_epoch_counter

def train_epoch_tsk(model, dataloader, lr = 5e-4, alpha=1, beta=1.0, device='cuda', info = 'training'):
    """Train for one epoch updating only regression parameters"""
    model.train()
    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.video_encoder.eval()
    
    optimizer_reg = torch.optim.Adam(model.get_regression_parameters(), lr = lr)
    # optimizer_reg = torch.optim.Adam(list(model.get_regression_parameters()) + list(model.get_simulation_parameters()), lr = lr)

    

    reg_loss_sum_epoch = 0.0
    within_epoch_counter = 0
    
    for batch in tqdm(dataloader, desc=info):
        within_epoch_counter += 1
        text = batch['text']
        audio = batch['audio']
        video = batch['video']
        labels = batch['label'].to(device)
        has_text = batch['has_text']
        has_audio = batch['has_audio']
        has_video = batch['has_video']
        
        # Forward pass
        predictions, _ = model(text, audio, video, has_text, has_audio, has_video)
        reg_loss = F.mse_loss(predictions, labels)

        # Accumulate and step regression optimizer
        reg_loss_value = reg_loss.item()
        reg_loss_sum_epoch += reg_loss_value
        
        optimizer_reg.zero_grad()
        reg_loss.backward()
        optimizer_reg.step()
    
    if within_epoch_counter == 0:
        return 0.0
    return reg_loss_sum_epoch / within_epoch_counter

def evaluate_sim(model, dataloader, device='cuda', info='evaluating'):
    """Evaluate the regression model"""
    model.eval()
    counter = 0.0
    sim_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc= info):
            counter =+ 1.0
            text = batch['text']
            audio = batch['audio']
            video = batch['video']
            labels = batch['label'].to(device)
            has_text = batch['has_text']
            has_audio = batch['has_audio']
            has_video = batch['has_video']
            
            # model.forward returns (predictions, sim_t, sim_a, sim_v)
            _, sim = model(text, audio, video, has_text, has_audio, has_video)
            sim_loss += sim.item()
            
    
    return sim_loss / counter

def evaluate(model, dataloader, device='cuda', info='evaluating'):
    """Evaluate the regression model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=info):
            text = batch['text']
            audio = batch['audio']
            video = batch['video']
            labels = batch['label'].to(device)
            has_text = batch['has_text']
            has_audio = batch['has_audio']
            has_video = batch['has_video']
            
            # model.forward returns (predictions, sim_t, sim_a, sim_v)
            predictions, _ = model(text, audio, video, has_text, has_audio, has_video)
            
            all_preds.extend(predictions.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    
    try:
        corr, _ = pearsonr(all_labels, all_preds)
    except Exception:
        corr = float('nan')
    
    binary_preds = [1 if p >= 0 else 0 for p in all_preds]
    binary_labels = [1 if l >= 0 else 0 for l in all_labels]
    binary_acc = sum(1 for p, l in zip(binary_preds, binary_labels) if p == l) / len(binary_labels) if binary_labels else 0.0
    
    return mse, mae, rmse, corr, binary_acc, all_preds, all_labels

def visualize_modality_gap(model, incomplete_dataloader, complete_dataloader, device, epoch, 
                           save_path='modality_gap_visualization_reg.png'):
    """
    Visualize modality gap between REAL and SIMULATED features for each modality.
    
    Args:
        model: The trained model
        incomplete_dataloader: Dataloader with missing modalities (current configuration)
        complete_dataloader: Dataloader with complete modalities (100% availability)
        device: cuda/cpu
        epoch: current epoch number
        save_path: where to save the visualization
    """
    model.eval()
    
    print(f"\nComplete dataloader has {len(complete_dataloader.dataset)} samples")
    print(f"Incomplete dataloader has {len(incomplete_dataloader.dataset)} samples")
    
    # Storage for real and simulated features
    real_text_features = []
    sim_text_features = []
    real_audio_features = []
    sim_audio_features = []
    real_video_features = []
    sim_video_features = []
    
    sample_indices_text = []  # Track which samples have missing text
    sample_indices_audio = []  # Track which samples have missing audio
    sample_indices_video = []  # Track which samples have missing video
    
    with torch.no_grad():
        # Iterate through both dataloaders simultaneously
        for incomplete_batch, complete_batch in tqdm(
            zip(incomplete_dataloader, complete_dataloader), 
            desc="Extracting features", 
            total=len(incomplete_dataloader),
            leave=False
        ):
            # Verify samples are aligned
            assert incomplete_batch['name'] == complete_batch['name'], \
                "Sample mismatch between complete and incomplete dataloaders!"
            
            batch_size = len(incomplete_batch['text'])
            
            # ========== Extract REAL features from COMPLETE data ==========
            text_complete = complete_batch['text']
            audio_complete = complete_batch['audio']
            video_complete = complete_batch['video']
            
            text_feat_real = model.text_encoder(text_complete)  # [batch_size, 768]
            audio_feat_real = model.audio_encoder(audio_complete)  # [batch_size, 768]
            video_feat_real = model.video_encoder(video_complete)  # [batch_size, 512]
            
            # ========== Extract SIMULATED features from INCOMPLETE data ==========
            text_incomplete = incomplete_batch['text']
            audio_incomplete = incomplete_batch['audio']
            video_incomplete = incomplete_batch['video']
            has_text_incomplete = incomplete_batch['has_text']
            has_audio_incomplete = incomplete_batch['has_audio']
            has_video_incomplete = incomplete_batch['has_video']
            
            # For each sample, check which modality is missing and simulate it
            for i in range(batch_size):
                t_avail = has_text_incomplete[i]
                a_avail = has_audio_incomplete[i]
                v_avail = has_video_incomplete[i]
                
                # ===== TEXT Simulation =====
                if not t_avail and a_avail and v_avail:
                    # Text is missing, audio and video available
                    audio_feat = model.audio_encoder([audio_incomplete[i]])
                    video_feat = model.video_encoder([video_incomplete[i]])
                    fused_av = model.fuse_av(audio_feat, video_feat)
                    sim_text = model.sim_av_t(fused_av)  # [1, 768]
                    
                    real_text_features.append(text_feat_real[i:i+1].cpu().numpy())
                    sim_text_features.append(sim_text.cpu().numpy())
                    sample_indices_text.append(len(real_text_features) - 1)
                
                elif not t_avail and a_avail and not v_avail:
                    # Only audio available, simulate text from audio
                    audio_feat = model.audio_encoder([audio_incomplete[i]])
                    sim_text = model.sim_a_t(audio_feat)
                    
                    real_text_features.append(text_feat_real[i:i+1].cpu().numpy())
                    sim_text_features.append(sim_text.cpu().numpy())
                    sample_indices_text.append(len(real_text_features) - 1)
                
                elif not t_avail and not a_avail and v_avail:
                    # Only video available, simulate text from video
                    video_feat = model.video_encoder([video_incomplete[i]])
                    sim_text = model.sim_v_t(video_feat)
                    
                    real_text_features.append(text_feat_real[i:i+1].cpu().numpy())
                    sim_text_features.append(sim_text.cpu().numpy())
                    sample_indices_text.append(len(real_text_features) - 1)
                
                # ===== AUDIO Simulation =====
                if not a_avail and t_avail and v_avail:
                    # Audio is missing, text and video available
                    text_feat = model.text_encoder([text_incomplete[i]])
                    video_feat = model.video_encoder([video_incomplete[i]])
                    fused_tv = model.fuse_tv(text_feat, video_feat)
                    sim_audio = model.sim_tv_a(fused_tv)  # [1, 768]
                    
                    real_audio_features.append(audio_feat_real[i:i+1].cpu().numpy())
                    sim_audio_features.append(sim_audio.cpu().numpy())
                    sample_indices_audio.append(len(real_audio_features) - 1)
                
                elif not a_avail and t_avail and not v_avail:
                    # Only text available, simulate audio from text
                    text_feat = model.text_encoder([text_incomplete[i]])
                    sim_audio = model.sim_t_a(text_feat)
                    
                    real_audio_features.append(audio_feat_real[i:i+1].cpu().numpy())
                    sim_audio_features.append(sim_audio.cpu().numpy())
                    sample_indices_audio.append(len(real_audio_features) - 1)
                
                elif not a_avail and not t_avail and v_avail:
                    # Only video available, simulate audio from video
                    video_feat = model.video_encoder([video_incomplete[i]])
                    sim_audio = model.sim_v_a(video_feat)
                    
                    real_audio_features.append(audio_feat_real[i:i+1].cpu().numpy())
                    sim_audio_features.append(sim_audio.cpu().numpy())
                    sample_indices_audio.append(len(real_audio_features) - 1)
                
                # ===== VIDEO Simulation =====
                if not v_avail and t_avail and a_avail:
                    # Video is missing, text and audio available
                    text_feat = model.text_encoder([text_incomplete[i]])
                    audio_feat = model.audio_encoder([audio_incomplete[i]])
                    fused_ta = model.fuse_ta(text_feat, audio_feat)
                    sim_video = model.sim_ta_v(fused_ta)  # [1, 512]
                    
                    real_video_features.append(video_feat_real[i:i+1].cpu().numpy())
                    sim_video_features.append(sim_video.cpu().numpy())
                    sample_indices_video.append(len(real_video_features) - 1)
                
                elif not v_avail and t_avail and not a_avail:
                    # Only text available, simulate video from text
                    text_feat = model.text_encoder([text_incomplete[i]])
                    sim_video = model.sim_t_v(text_feat)
                    
                    real_video_features.append(video_feat_real[i:i+1].cpu().numpy())
                    sim_video_features.append(sim_video.cpu().numpy())
                    sample_indices_video.append(len(real_video_features) - 1)
                
                elif not v_avail and not t_avail and a_avail:
                    # Only audio available, simulate video from audio
                    audio_feat = model.audio_encoder([audio_incomplete[i]])
                    sim_video = model.sim_a_v(audio_feat)
                    
                    real_video_features.append(video_feat_real[i:i+1].cpu().numpy())
                    sim_video_features.append(sim_video.cpu().numpy())
                    sample_indices_video.append(len(real_video_features) - 1)
    
    # Check if we have any samples to visualize
    if len(real_text_features) == 0:
        print("⚠️  No missing TEXT samples found - skipping TEXT visualization")
    if len(real_audio_features) == 0:
        print("⚠️  No missing AUDIO samples found - skipping AUDIO visualization")
    if len(real_video_features) == 0:
        print("⚠️  No missing VIDEO samples found - skipping VIDEO visualization")
    
    # Concatenate all samples
    if len(real_text_features) > 0:
        real_text_features = np.vstack(real_text_features)
        sim_text_features = np.vstack(sim_text_features)
    if len(real_audio_features) > 0:
        real_audio_features = np.vstack(real_audio_features)
        sim_audio_features = np.vstack(sim_audio_features)
    if len(real_video_features) > 0:
        real_video_features = np.vstack(real_video_features)
        sim_video_features = np.vstack(sim_video_features)
    
    print(f"\nCollected features:")
    print(f"  Text:  {len(real_text_features)} samples with missing text")
    print(f"  Audio: {len(real_audio_features)} samples with missing audio")
    print(f"  Video: {len(real_video_features)} samples with missing video")
    
    # Limit samples for visualization
    max_samples = 500
    
    # Create visualization
    num_plots = sum([
        len(real_text_features) > 0,
        len(real_audio_features) > 0,
        len(real_video_features) > 0
    ])
    
    if num_plots == 0:
        print("⚠️  No missing modalities to visualize!")
        return
    
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'Real vs Simulated Features - Epoch {epoch}', fontsize=18, fontweight='bold')
    
    plot_idx = 0
    
    # TEXT: Real vs Simulated
    if len(real_text_features) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        
        n_samples = min(max_samples, len(real_text_features))
        real_text_subset = real_text_features[:n_samples]
        sim_text_subset = sim_text_features[:n_samples]
        
        all_text = np.vstack([real_text_subset, sim_text_subset])
        print(f"\nApplying t-SNE for TEXT features (epoch {epoch})...")
        tsne_text = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        text_2d = tsne_text.fit_transform(all_text)
        n = len(real_text_subset)
        real_text_2d = text_2d[:n]
        sim_text_2d = text_2d[n:]
        
        ax.scatter(real_text_2d[:, 0], real_text_2d[:, 1], 
                  c='blue', label='Real Text', alpha=0.6, s=50, marker='o', edgecolors='black', linewidths=0.5)
        ax.scatter(sim_text_2d[:, 0], sim_text_2d[:, 1], 
                  c='red', label='Simulated Text', alpha=0.6, s=50, marker='^', edgecolors='black', linewidths=0.5)
        ax.set_title(f'Text: Real vs Simulated (n={n_samples})', fontweight='bold', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # AUDIO: Real vs Simulated
    if len(real_audio_features) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        
        n_samples = min(max_samples, len(real_audio_features))
        real_audio_subset = real_audio_features[:n_samples]
        sim_audio_subset = sim_audio_features[:n_samples]
        
        all_audio = np.vstack([real_audio_subset, sim_audio_subset])
        print(f"Applying t-SNE for AUDIO features (epoch {epoch})...")
        tsne_audio = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        audio_2d = tsne_audio.fit_transform(all_audio)
        n = len(real_audio_subset)
        real_audio_2d = audio_2d[:n]
        sim_audio_2d = audio_2d[n:]
        
        ax.scatter(real_audio_2d[:, 0], real_audio_2d[:, 1], 
                  c='green', label='Real Audio', alpha=0.6, s=50, marker='o', edgecolors='black', linewidths=0.5)
        ax.scatter(sim_audio_2d[:, 0], sim_audio_2d[:, 1], 
                  c='red', label='Simulated Audio', alpha=0.6, s=50, marker='^', edgecolors='black', linewidths=0.5)
        ax.set_title(f'Audio: Real vs Simulated (n={n_samples})', fontweight='bold', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # VIDEO: Real vs Simulated
    if len(real_video_features) > 0:
        ax = axes[plot_idx]
        plot_idx += 1
        
        n_samples = min(max_samples, len(real_video_features))
        real_video_subset = real_video_features[:n_samples]
        sim_video_subset = sim_video_features[:n_samples]
        
        all_video = np.vstack([real_video_subset, sim_video_subset])
        print(f"Applying t-SNE for VIDEO features (epoch {epoch})...")
        tsne_video = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        video_2d = tsne_video.fit_transform(all_video)
        n = len(real_video_subset)
        real_video_2d = video_2d[:n]
        sim_video_2d = video_2d[n:]
        
        ax.scatter(real_video_2d[:, 0], real_video_2d[:, 1], 
                  c='purple', label='Real Video', alpha=0.6, s=50, marker='o', edgecolors='black', linewidths=0.5)
        ax.scatter(sim_video_2d[:, 0], sim_video_2d[:, 1], 
                  c='red', label='Simulated Video', alpha=0.6, s=50, marker='^', edgecolors='black', linewidths=0.5)
        ax.set_title(f'Video: Real vs Simulated (n={n_samples})', fontweight='bold', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved modality gap visualization to {save_path}")
    
    # Calculate and print average distances
    from scipy.spatial.distance import cdist
    
    print(f"\nAverage Euclidean distances (Real vs Simulated):")
    if len(real_text_features) > 0:
        text_dist = np.mean(np.linalg.norm(real_text_features - sim_text_features, axis=1))
        print(f"  Text:  {text_dist:.4f}")
    if len(real_audio_features) > 0:
        audio_dist = np.mean(np.linalg.norm(real_audio_features - sim_audio_features, axis=1))
        print(f"  Audio: {audio_dist:.4f}")
    if len(real_video_features) > 0:
        video_dist = np.mean(np.linalg.norm(real_video_features - sim_video_features, axis=1))
        print(f"  Video: {video_dist:.4f}")

import json
import os
from sklearn.metrics import f1_score

def save_results_to_json(results_dict, filepath):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results_dict, indent=4, fp=f)
    print(f"✓ Saved results to {filepath}")


def calculate_f1_scores(all_preds, all_labels):
    """Calculate F1 micro and macro scores"""
    # Convert continuous predictions to binary (positive/negative sentiment)
    binary_preds = [1 if p >= 0 else 0 for p in all_preds]
    binary_labels = [1 if l >= 0 else 0 for l in all_labels]
    
    f1_micro = f1_score(binary_labels, binary_preds, average='micro')
    f1_macro = f1_score(binary_labels, binary_preds, average='macro')
    
    return f1_micro, f1_macro


from typing import List, Optional
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(missing_configs: Optional[List[str]] = None) -> None:
    audio_dir = dsspath + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Audio/WAV_16000/Segmented"
    video_dir = dsspath + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Video/Segmented"
    text_dir = dsspath + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Transcript/Segmented"
    split_file = dsspath + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/mosi_splits-70train.json"

    num_my_workers = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if missing_configs is None:
        missing_configs = [         
        "100_text_100_audio_100_video",
        "100_text_20_audio_20_video",    
        "20_text_100_audio_20_video", 
        "20_text_20_audio_100_video",          
        "20_text_100_audio_100_video",
        "100_text_100_audio_20_video",  
        "complex_20_20_20_10_10_10_10", 
        "100_text_20_audio_100_video"
        ]
    elif len(missing_configs[-1]) < 3:
        missing_configs.pop()
    
    if missing_configs[-1] == "cpu":
        device = 'cpu'
        
    print(f"missing-config:s: {missing_configs}, device: {device}")
    
    # ========== EXPERIMENT CONFIGURATION ==========   
    num_runs = 10  # Number of independent runs
    complete_config = "100_text_100_audio_100_video"

    my_seeds = [
        42, 123, 256, 512, 1024,
        2048, 3141, 5000, 7777, 9999,
        12345, 54321, 11111, 22222, 33333,
        44444, 55555, 66666, 77777, 88888,
        99999, 13579, 24680, 31415, 27182,
        16180, 86753, 10101, 20202, 30303
    ]  
    
    batch_size = 8
    num_epochs_sim = 5
    num_epochs_tsk = 15    
    lr = 5e-4
    alpha = 1.0
    beta = 1.0
    
    # Create results directory
    results_dir = f"XCLP_WLM-s{num_epochs_sim}-e{num_epochs_tsk}-r{num_runs}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"MULTI-CONFIG MULTI-RUN EXPERIMENT")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Configurations: {len(missing_configs)}")
    print(f"Runs per config: {num_runs}")
    print(f"Total experiments: {len(missing_configs) * num_runs}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")
    
    # Analyze dataset once
    #analyze_mosi_dataset(audio_dir, video_dir, text_dir, split_file)
    
    # ========== MAIN EXPERIMENT LOOP ==========
    all_experiments_results = {}
    
    for config_idx, missing_config in enumerate(missing_configs):
        print(f"\n{'#'*80}")
        print(f"# CONFIG {config_idx+1}/{len(missing_configs)}: {missing_config}")
        print(f"{'#'*80}\n")
        
        config_results = []
        f1_micro_total = 0.0
        counter_run = 0.0
        info_to_print = f"{missing_config}"
        for run_idx in range(num_runs):
            if counter_run > 0:
                info_to_print = f"{missing_config}, ave till {run_idx}-th run: {f1_micro_total / (counter_run):.4f}"
            counter_run += 1.0
            print(f"\n{'='*80}")
            print(f"RUN {run_idx+1}/{num_runs} | CONFIG: {missing_config}")
            print(f"{'='*80}\n")
            
            # Set different seed for each run
            run_seed = my_seeds[run_idx]
            set_seed(run_seed)

            # Create datasets
            train_dataset = MOSIDatasetRegression(
                audio_dir, video_dir, text_dir, split_file,
                split='train', missing_config=missing_config, seed=run_seed
            )
            val_dataset = MOSIDatasetRegression(
                audio_dir, video_dir, text_dir, split_file,
                split='val', missing_config=missing_config, seed=run_seed
            )
            test_dataset = MOSIDatasetRegression(
                audio_dir, video_dir, text_dir, split_file,
                split='test', missing_config=missing_config, seed=run_seed
            )
            
            complete_val_dataset = MOSIDatasetRegression(
                audio_dir, video_dir, text_dir, split_file,
                split='val', missing_config=complete_config, seed=run_seed
            )
            complete_test_dataset = MOSIDatasetRegression(
                audio_dir, video_dir, text_dir, split_file,
                split='test', missing_config=complete_config, seed=run_seed
            )
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    collate_fn=collate_fn, num_workers=num_my_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                  collate_fn=collate_fn, num_workers=num_my_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                   collate_fn=collate_fn, num_workers=num_my_workers)
            
            complete_val_loader = DataLoader(complete_val_dataset, batch_size=batch_size, 
                                           shuffle=False, collate_fn=collate_fn, num_workers=num_my_workers)
            complete_test_loader = DataLoader(complete_test_dataset, batch_size=batch_size, 
                                            shuffle=False, collate_fn=collate_fn, num_workers=num_my_workers)
            
            # Create model
            model = Net().to(device)
            
            print("\n" + "="*80)
            print("PHASE 1: SIMULATION TRAINING")
            print("="*80 + "\n")
            
            # ========== SIMULATION TRAINING ==========
            best_sim_loss = float('inf')
            best_loss_sum = float('inf')
            import copy
            best_model_sim = None
            
            for epoch in range(num_epochs_sim):
                print(f"\nSimulation Epoch {epoch+1}/{num_epochs_sim}")
                print("-" * 80)
                
                # Train simulation
                sim_loss = train_epoch_sim(
                    model, train_loader, lr, alpha, beta, device, info_to_print
                )
                print(f"Train - Total: {sim_loss:.4f}")
                
                # Validate simulation
                val_sim_loss = evaluate_sim(model, val_loader, device, info_to_print)               
                
                if val_sim_loss < best_sim_loss:
                    best_sim_loss = val_sim_loss
                    best_model_sim = copy.deepcopy(model.state_dict())
                    print("✓ New best simulation model (lower max loss)")
                
                print(f"Val - Total: {val_sim_loss:.4f}")
            
            # Load best simulation model
            if best_model_sim is not None:
                model.load_state_dict(best_model_sim)
                print(f"\n✓ Loaded best simulation model (max_loss={best_sim_loss:.4f})")
            
            # Save simulation model
            sim_model_path = os.path.join(
                results_dir, 
                f"sim-e_{num_epochs_sim}-lr_{lr}-{missing_config}-{run_idx+1}.pt"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': missing_config,
                'run': run_idx + 1,
                'num_epochs_sim': num_epochs_sim,
                'lr': lr,
                'best_sim_loss': best_sim_loss
            }, sim_model_path)
            print(f"✓ Saved simulation model to {sim_model_path}")
            
            # Visualize after simulation
            """ vis_path_sim = os.path.join(
                results_dir,
                f"vis_sim-e_{num_epochs_sim}-lr_{lr}-{missing_config}-{run_idx+1}.png"
            )
            visualize_modality_gap(model, val_loader, complete_val_loader, device, 
                                  epoch=f'sim-{num_epochs_sim}', save_path=vis_path_sim) """
            
            print("\n" + "="*80)
            print("PHASE 2: TASK TRAINING")
            print("="*80 + "\n")
            
            # ========== TASK TRAINING ==========
            best_binary_acc = 0.0
            best_model_task = None
            
            for epoch in range(num_epochs_tsk):
                print(f"\nTask Epoch {epoch+1}/{num_epochs_tsk}")
                print("-" * 80)
                
                # Train task
                reg_loss = train_epoch_tsk(model, train_loader, lr, alpha, beta, device, info_to_print)
                
                # Validate task
                mse, mae, rmse, corr, binary_acc, _, _ = evaluate(model, val_loader, device, info_to_print)
                
                if binary_acc > best_binary_acc:
                    best_binary_acc = binary_acc
                    best_model_task = copy.deepcopy(model.state_dict())
                    print("✓ New best task model")
                
                print(f"Val - MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | "
                      f"Corr: {corr:.4f} | BinAcc: {binary_acc:.4f}")
            
            # Load best task model
            if best_model_task is not None:
                model.load_state_dict(best_model_task)
                print(f"\n✓ Loaded best task model (binary_acc={best_binary_acc:.4f})")
            
            # Save final model
            final_model_path = os.path.join(
                results_dir,
                f"se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}-{missing_config}-{run_idx+1}.pt"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': missing_config,
                'run': run_idx + 1,
                'num_epochs_sim': num_epochs_sim,
                'num_epochs_tsk': num_epochs_tsk,
                'lr': lr,
                'best_binary_acc': best_binary_acc
            }, final_model_path)
            print(f"✓ Saved final model to {final_model_path}")
            
            # Visualize after task training
            vis_path_final = os.path.join(
                results_dir,
                f"vis_final-se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}-{missing_config}-{run_idx+1}.png"
            )
            visualize_modality_gap(model, test_loader, complete_test_loader, device,
                                  epoch='final', save_path=vis_path_final)
            
            print("\n" + "="*80)
            print("PHASE 3: TEST EVALUATION")
            print("="*80 + "\n")
            
            # ========== TEST EVALUATION ==========
            test_mse, test_mae, test_rmse, test_corr, test_binary_acc, test_preds, test_labels = \
                evaluate(model, test_loader, device)
            
            # Calculate F1 scores
            f1_micro, f1_macro = calculate_f1_scores(test_preds, test_labels)

            f1_micro_total += f1_micro
            
            # Calculate tolerance accuracies
            tolerance_05 = sum([1 for p, l in zip(test_preds, test_labels) 
                              if abs(p - l) < 0.5]) / len(test_labels)
            tolerance_10 = sum([1 for p, l in zip(test_preds, test_labels) 
                              if abs(p - l) < 1.0]) / len(test_labels)
            
            print(f"Test Results:")
            print(f"  MSE:               {test_mse:.4f}")
            print(f"  MAE:               {test_mae:.4f}")
            print(f"  RMSE:              {test_rmse:.4f}")
            print(f"  Pearson Corr:      {test_corr:.4f}")
            print(f"  Binary Accuracy:   {test_binary_acc:.4f}")
            print(f"  F1-Micro:          {f1_micro:.4f}")
            print(f"  F1-Macro:          {f1_macro:.4f}")
            print(f"  Acc@0.5:           {tolerance_05:.4f}")
            print(f"  Acc@1.0:           {tolerance_10:.4f}")
            
            # Save results to JSON
            results = {
                "config": missing_config,
                "run": run_idx + 1,
                "seed": run_seed,
                "num_epochs_sim": num_epochs_sim,
                "num_epochs_tsk": num_epochs_tsk,
                "lr": lr,
                "batch_size": batch_size,
                "test_metrics": {
                    "MSE": float(test_mse),
                    "MAE": float(test_mae),
                    "RMSE": float(test_rmse),
                    "Pearson_Corr": float(test_corr),
                    "Binary_Accuracy": float(test_binary_acc),
                    "F1_Micro": float(f1_micro),
                    "F1_Macro": float(f1_macro),
                    "Acc@0.5": float(tolerance_05),
                    "Acc@1.0": float(tolerance_10)
                }
            }
            
            results_path = os.path.join(
                results_dir,
                f"se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}-{missing_config}-{run_idx+1}.json"
            )
            save_results_to_json(results, results_path)
            
            config_results.append(results)
            
            # Create prediction scatter plot
            scatter_path = os.path.join(
                results_dir,
                f"scatter-se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}-{missing_config}-{run_idx+1}.png"
            )
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(test_labels, test_preds, alpha=0.5, s=50)
            
            min_val = min(min(test_labels), min(test_preds))
            max_val = max(max(test_labels), max(test_preds))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                   label='Perfect Prediction')
            
            ax.set_xlabel('True Sentiment Score', fontsize=14)
            ax.set_ylabel('Predicted Sentiment Score', fontsize=14)
            ax.set_title(f'Run {run_idx+1} | MAE: {test_mae:.4f}, Corr: {test_corr:.4f}', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved scatter plot to {scatter_path}")
            
            print(f"\n{'='*80}")
            print(f"COMPLETED RUN {run_idx+1}/{num_runs} FOR CONFIG {missing_config}")
            print(f"{'='*80}\n")
        
       
        # all_experiments_results[missing_config] = config_results        
       
        # print(f"\n{'='*80}")
        # print(f"AGGREGATE RESULTS FOR CONFIG: {missing_config}")
        # print(f"{'='*80}")
        
        # avg_mse = np.mean([r['MSE'] for r in config_results])
        # std_mse = np.std([r['MSE'] for r in config_results])
        # avg_mae = np.mean([r['MAE'] for r in config_results])
        # std_mae = np.std([r['MAE'] for r in config_results])
        # avg_rmse = np.mean([r['RMSE'] for r in config_results])
        # std_rmse = np.std([r['RMSE'] for r in config_results])
        # avg_corr = np.mean([r['Pearson_Corr'] for r in config_results])
        # std_corr = np.std([r['Pearson_Corr'] for r in config_results])
        # avg_binacc = np.mean([r['Binary_Accuracy'] for r in config_results])
        # std_binacc = np.std([r['Binary_Accuracy'] for r in config_results])
        # avg_f1_micro = np.mean([r['F1_Micro'] for r in config_results])
        # std_f1_micro = np.std([r['F1_Micro'] for r in config_results])
        # avg_f1_macro = np.mean([r['F1_Macro'] for r in config_results])
        # std_f1_macro = np.std([r['F1_Macro'] for r in config_results])
        
        # print(f"MSE:        {avg_mse:.4f} ± {std_mse:.4f}")
        # print(f"MAE:        {avg_mae:.4f} ± {std_mae:.4f}")
        # print(f"RMSE:       {avg_rmse:.4f} ± {std_rmse:.4f}")
        # print(f"Corr:       {avg_corr:.4f} ± {std_corr:.4f}")
        # print(f"BinAcc:     {avg_binacc:.4f} ± {std_binacc:.4f}")
        # print(f"F1-Micro:   {avg_f1_micro:.4f} ± {std_f1_micro:.4f}")
        # print(f"F1-Macro:   {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
        # print(f"{'='*80}\n")        
        
        # aggregate_results = {
        #     "config": missing_config,
        #     "num_runs": num_runs,
        #     "num_epochs_sim": num_epochs_sim,
        #     "num_epochs_tsk": num_epochs_tsk,
        #     "lr": lr,
        #     "metrics": {
        #         "MSE": {"mean": float(avg_mse), "std": float(std_mse)},
        #         "MAE": {"mean": float(avg_mae), "std": float(std_mae)},
        #         "RMSE": {"mean": float(avg_rmse), "std": float(std_rmse)},
        #         "Pearson_Corr": {"mean": float(avg_corr), "std": float(std_corr)},
        #         "Binary_Accuracy": {"mean": float(avg_binacc), "std": float(std_binacc)},
        #         "F1_Micro": {"mean": float(avg_f1_micro), "std": float(std_f1_micro)},
        #         "F1_Macro": {"mean": float(avg_f1_macro), "std": float(std_f1_macro)}
        #     },
        #     "individual_runs": config_results
        # }
        
        # aggregate_path = os.path.join(
        #     results_dir,
        #     f"aggregate-se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}-{missing_config}.json"
        # )
        # save_results_to_json(aggregate_results, aggregate_path)
    
    # ========== FINAL SUMMARY ==========
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT COMPLETE!")
    print(f"{'#'*80}\n")
    
    print(f"Total configurations: {len(missing_configs)}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total experiments completed: {len(missing_configs) * num_runs}")
    print(f"Results saved in: {results_dir}/")
    
    # Create final summary
    summary = {
        "experiment_info": {
            "num_configs": len(missing_configs),
            "num_runs": num_runs,
            "total_experiments": len(missing_configs) * num_runs,
            "num_epochs_sim": num_epochs_sim,
            "num_epochs_tsk": num_epochs_tsk,
            "lr": lr,
            "batch_size": batch_size,
            "device": device
        },
        "configs": missing_configs,
        "results": all_experiments_results
    }
    
    summary_path = os.path.join(results_dir, f"SUMMARY-se_{num_epochs_sim}-te_{num_epochs_tsk}-lr_{lr}.json")
    save_results_to_json(summary, summary_path)
    
    print(f"\n✓ Final summary saved to: {summary_path}")
    print(f"\n{'#'*80}\n")

import sys
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided configs via terminal → use them
        main(sys.argv[1:])
    else:
        # No arguments → use the default list
        main()
