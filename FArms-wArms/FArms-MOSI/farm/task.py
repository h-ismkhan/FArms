import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import BertModel, BertTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from scipy.stats import pearsonr
from typing import Tuple
import warnings
import cv2
import torchaudio

import copy
from tqdm import tqdm


warnings.filterwarnings('ignore')

from farm.mosi_reg import MOSIDatasetRegression

# Dataset paths
_dpath = "/home/_Dataset/"
AUDIO_DIR = _dpath + "Raw - CMU Multimodal Opinion Sentiment Intensity/Audio/WAV_16000/Segmented"
VIDEO_DIR = _dpath + "Raw - CMU Multimodal Opinion Sentiment Intensity/Video/Segmented"
TEXT_DIR = _dpath + "Raw - CMU Multimodal Opinion Sentiment Intensity/Transcript/Segmented"
SPLIT_FILE = _dpath + "Raw - CMU Multimodal Opinion Sentiment Intensity/mosi_splits-70train.json"


# ============================================================================
# FROZEN ENCODERS
#
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


# ============================================================================
# DATA LOADING
# ============================================================================

def custom_collate_fn(batch):
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


def load_data(seed: int, partition_id: int, num_partitions: int, missing_config: str):
    """Load federated partition of MOSI dataset"""
    import json
    
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    
    train_samples = splits['train']
    
    # Partition the training data
    partition_size = len(train_samples) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(train_samples)
    partition_samples = train_samples[start_idx:end_idx]
    
    # Use 20% for local validation
    val_size = int(0.2 * len(partition_samples))
    val_samples = partition_samples[:val_size]
    train_samples_final = partition_samples[val_size:]
    
    # Create temporary split files
    temp_train_split = {'train': train_samples_final, 'val': val_samples, 'test': splits['test']}
    temp_val_split = {'train': train_samples_final, 'val': val_samples, 'test': splits['test']}
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_train_split, f)
        temp_split_file = f.name
    
    train_dataset = MOSIDatasetRegression(
        AUDIO_DIR, VIDEO_DIR, TEXT_DIR, temp_split_file,
        split='train', missing_config=missing_config, seed=seed
    )
    
    val_dataset = MOSIDatasetRegression(
        AUDIO_DIR, VIDEO_DIR, TEXT_DIR, temp_split_file,
        split='val', missing_config=missing_config, seed=seed
    )
    
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                            collate_fn=custom_collate_fn, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                          collate_fn=custom_collate_fn, num_workers=0)
    
    return trainloader, valloader


def load_test_data(seed: int, missing_config: str):
    """Load complete test set for final evaluation"""
    test_dataset = MOSIDatasetRegression(
        AUDIO_DIR, VIDEO_DIR, TEXT_DIR, SPLIT_FILE,
        split='test', missing_config=missing_config, seed=seed
    )
    
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                           collate_fn=custom_collate_fn, num_workers=0)
    
    return testloader

# ============================================================================
# TRAINING AND EVALUATION - SEPARATE OPTIMIZERS
# ============================================================================
def train_sim(model, trainloader, lr, device, info_to_print):
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

    for batch in trainloader:
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

def train_task(model, trainloader, lr, device, missing_config):
    #we never need missing_config here, just for debug
    model.train()
    model.text_encoder.eval()
    model.audio_encoder.eval()
    model.video_encoder.eval()
    
    # optimizer_reg = torch.optim.Adam(model.get_regression_parameters(), lr = lr)
    optimizer_reg = torch.optim.Adam(list(model.get_regression_parameters()) + list(model.get_simulation_parameters()), lr = lr)

    reg_loss_sum_epoch = 0.0
    within_epoch_counter = 0
    for batch in trainloader:
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


def test(net, testloader, device):
    """Evaluate the model"""
    net = net.to(device)
    net.eval()
    
    loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in testloader:
            text = batch['text']
            audio = batch['audio']
            video = batch['video']
            labels = batch['label'].to(device)
            has_text = batch['has_text']
            has_audio = batch['has_audio']
            has_video = batch['has_video']
            
            predictions, _  = net(text, audio, video, has_text, has_audio, has_video)
            loss += F.mse_loss(predictions, labels).item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = loss / len(testloader)
    mae = mean_absolute_error(all_labels, all_preds)
    
    return loss, mae


def compute_classification_metrics(predictions, labels):
    """
    Compute classification metrics by converting regression outputs to binary classes.
    Positive (>=0) -> 1, Negative (<0) -> 0
    """
    # Convert to binary classes
    binary_preds = np.array([1 if p >= 0 else 0 for p in predictions])
    binary_labels = np.array([1 if l >= 0 else 0 for l in labels])
    
    # Calculate binary accuracy
    binary_acc = np.mean(binary_preds == binary_labels)
    
    # Calculate precision, recall, f1 for both classes
    precision, recall, f1, support = precision_recall_fscore_support(
        binary_labels, binary_preds, average=None, labels=[0, 1], zero_division=0
    )
    
    # Micro-averaged F1 (same as accuracy for binary classification)
    f1_micro = np.mean(binary_preds == binary_labels)
    
    # Macro-averaged F1
    f1_macro = np.mean(f1)
    
    # Sample-averaged F1 (for binary classification, same as micro)
    f1_sample = f1_micro
    
    return {
        'binary_accuracy': float(binary_acc),
        'precision_negative': float(precision[0]),
        'precision_positive': float(precision[1]),
        'recall_negative': float(recall[0]),
        'recall_positive': float(recall[1]),
        'f1_negative': float(f1[0]),
        'f1_positive': float(f1[1]),
        'f1_micro': float(f1_micro),
        'f1_macro': float(f1_macro),
        'f1_sample': float(f1_sample)
    }


def test_final(net, testloader, device, print_result = True):
    """Final test evaluation with detailed metrics"""
    net = net.to(device)
    net.eval()
    
    if print_result:
        print(f"\n{'#'*80}")
        print(f"# FINAL TEST SET EVALUATION")
        print(f"{'#'*80}\n")
    
    loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in testloader:
            text = batch['text']
            audio = batch['audio']
            video = batch['video']
            labels = batch['label'].to(device)
            has_text = batch['has_text']
            has_audio = batch['has_audio']
            has_video = batch['has_video']
            
            predictions, _  = net(text, audio, video, has_text, has_audio, has_video)
            loss += F.mse_loss(predictions, labels).item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Regression metrics
    loss = loss / len(testloader)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    # Calculate Pearson correlation
    corr, _ = pearsonr(all_labels, all_preds)
    
    # Classification metrics
    classification_metrics = compute_classification_metrics(all_preds, all_labels)
    
    if print_result:
        print(f"\n{'='*80}")
        print(f"FINAL TEST SET RESULTS - REGRESSION")
        print(f"{'='*80}")
        print(f"  MSE:               {loss:.4f}")
        print(f"  MAE:               {mae:.4f}")
        print(f"  RMSE:              {rmse:.4f}")
        print(f"  Pearson Corr:      {corr:.4f}")
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION METRICS (Binary: Positive>=0, Negative<0)")
        print(f"{'='*80}")
        print(f"  Binary Accuracy:   {classification_metrics['binary_accuracy']:.4f}")
        print(f"  Precision (Neg):   {classification_metrics['precision_negative']:.4f}")
        print(f"  Precision (Pos):   {classification_metrics['precision_positive']:.4f}")
        print(f"  Recall (Neg):      {classification_metrics['recall_negative']:.4f}")
        print(f"  Recall (Pos):      {classification_metrics['recall_positive']:.4f}")
        print(f"  F1 (Neg):          {classification_metrics['f1_negative']:.4f}")
        print(f"  F1 (Pos):          {classification_metrics['f1_positive']:.4f}")
        print(f"  F1-Micro:          {classification_metrics['f1_micro']:.4f}")
        print(f"  F1-Macro:          {classification_metrics['f1_macro']:.4f}")
        print(f"  F1-Sample:         {classification_metrics['f1_sample']:.4f}")
        print(f"{'='*80}\n")
    
    return {
        'mse': float(loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson_corr': float(corr),
        **classification_metrics
    }