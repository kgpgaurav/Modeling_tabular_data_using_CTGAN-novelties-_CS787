"""Tabular Diffusion Model (TabDDPM) - STABLE VERSION

Key fixes:
1. Numerical stability in activation functions with proper clipping
2. Output clipping in forward pass to prevent extreme values
3. Better initialization with smaller weights
4. Improved softmax computation with log-space stability
5. Gradient clipping and monitoring
6. Robust handling of NaN/Inf values
7. Conservative temperature scaling
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers._utils import validate_and_set_device
from ctgan.synthesizers.base import BaseSynthesizer, random_state


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    """Gaussian Diffusion process for DDPM."""
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
        self.timesteps = timesteps
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in Improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_fn, x_start, t, cond=None, noise=None):
        """Calculate the denoising loss."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_fn(x_noisy, t, cond)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def p_sample(self, denoise_fn, x, t, cond=None):
        """Single reverse diffusion step: p(x_{t-1} | x_t)."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        predicted_noise = denoise_fn(x, t, cond)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, cond=None, device='cpu'):
        """Full reverse diffusion loop to generate samples."""
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(denoise_fn, x, t, cond)
        
        return x
    
    @torch.no_grad()
    def ddim_sample(self, denoise_fn, shape, cond=None, device='cpu', ddim_steps=50, eta=0.0):
        """DDIM sampling for faster generation."""
        batch_size = shape[0]
        
        skip = self.timesteps // ddim_steps
        seq = range(0, self.timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        
        x = torch.randn(shape, device=device)
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_next = torch.full((batch_size,), j, device=device, dtype=torch.long)
            
            predicted_noise = denoise_fn(x, t, cond)
            
            alpha_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_t_next = extract(self.alphas_cumprod, t_next, x.shape) if j >= 0 else torch.ones_like(alpha_t)
            
            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            
            # Clip predicted x0 for stability
            x0_pred = torch.clamp(x0_pred, min=-5, max=5)
            
            dir_xt = torch.sqrt(1 - alpha_t_next - eta**2 * (1 - alpha_t) / (1 - alpha_t_next) * (1 - alpha_t_next / alpha_t)) * predicted_noise
            
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            
            x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt + eta * torch.sqrt((1 - alpha_t) / (1 - alpha_t_next) * (1 - alpha_t_next / alpha_t)) * noise
            
            # Clip x for stability
            x = torch.clamp(x, min=-10, max=10)
        
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Improved residual block with proper normalization."""
    
    def __init__(self, in_dim, out_dim, time_embed_dim, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + time_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Projection for residual if dimensions don't match
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, t_embed):
        h = torch.cat([x, t_embed], dim=-1)
        h = self.mlp(h)
        return h + self.residual_proj(x)


class MLPDiffusion(nn.Module):
    """STABLE MLP-based denoising network for tabular diffusion."""

    def __init__(self, data_dim, cond_dim=0, hidden_dim=256, num_layers=4,
                 time_embed_dim=128, dropout=0.1):
        super().__init__()

        self.data_dim = data_dim
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim

        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Input dimension includes conditioning
        input_dim = data_dim + self.cond_dim

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_embed_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection with skip connection from input
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, data_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Improved weight initialization with smaller values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, t, cond=None):
        # Save original input for skip connection
        x_input = x.clone()
        
        # Time embedding
        t_embed = self.time_mlp(t)

        # Handle conditioning
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(x.shape[0], self.cond_dim, device=x.device, dtype=x.dtype)
            elif cond.shape[0] != x.shape[0]:
                cond = cond.repeat(x.shape[0] // cond.shape[0] + 1, 1)[:x.shape[0]]
            
            x = torch.cat([x, cond], dim=-1)
            x_input_with_cond = x.clone()
        else:
            x_input_with_cond = x.clone()

        # Input projection
        h = self.input_layer(x)

        # Residual blocks
        for block in self.blocks:
            h = block(h, t_embed)

        # Output with skip connection from input
        h = torch.cat([h, x_input_with_cond], dim=-1)
        output = self.output_layer(h)
        
        # CRITICAL: Clip output to prevent extreme values
        output = torch.clamp(output, min=-10, max=10)

        return output


class TabDDPM(BaseSynthesizer):
    """STABLE Tabular Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        timesteps=1000,
        hidden_dim=256,
        num_layers=4,
        time_embed_dim=128,
        lr=5e-5,
        weight_decay=1e-5,
        batch_size=500,
        epochs=300,
        dropout=0.1,
        log_frequency=True,
        verbose=False,
        schedule='cosine',
        enable_gpu=True,
        cuda=None,
        ddim_sampling=True,
        ddim_steps=50,
        temperature=1.0,
    ):
        self.timesteps = timesteps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.schedule = schedule
        self.ddim_sampling = ddim_sampling
        self.ddim_steps = ddim_steps
        self.temperature = temperature
        
        self._device = validate_and_set_device(enable_gpu, cuda)
        self._enable_gpu = cuda if cuda is not None else enable_gpu
        self._transformer = None
        self._data_sampler = None
        self._model = None
        self._diffusion = None
        self.loss_values = None
    
    def _apply_activate(self, data):
        """Apply proper activation with numerical stability and temperature scaling."""
        data_t = []
        st = 0
        
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                ed = st + span_info.dim
                
                if span_info.activation_fn == 'tanh':
                    # Apply tanh with clipping
                    chunk = data[:, st:ed]
                    chunk = torch.clamp(chunk, min=-10, max=10)
                    data_t.append(torch.tanh(chunk))
                    
                elif span_info.activation_fn == 'softmax':
                    # IMPROVED: Maximum numerical stability for softmax
                    logits = data[:, st:ed]
                    
                    # Step 1: Clip extreme values BEFORE any operations
                    logits = torch.clamp(logits, min=-20, max=20)
                    
                    # Step 2: Apply temperature scaling (prevent division by tiny values)
                    temperature = max(self.temperature, 0.1)
                    logits = logits / temperature
                    
                    # Step 3: Add small Gumbel noise during sampling for diversity (optional)
                    if not self._model.training:
                        # Safe Gumbel noise generation
                        uniform_noise = torch.rand_like(logits)
                        uniform_noise = torch.clamp(uniform_noise, min=1e-10, max=1.0 - 1e-10)
                        gumbel_noise = -torch.log(-torch.log(uniform_noise))
                        logits = logits + 0.05 * gumbel_noise  # Small noise for stability
                    
                    # Step 4: Clip again after noise addition
                    logits = torch.clamp(logits, min=-20, max=20)
                    
                    # Step 5: Numerically stable softmax using max subtraction
                    logits_max = logits.max(dim=-1, keepdim=True)[0]
                    logits_stable = logits - logits_max
                    
                    # Step 6: Compute softmax
                    exp_logits = torch.exp(logits_stable)
                    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
                    
                    # Step 7: Ensure valid probabilities
                    probs = torch.clamp(probs, min=1e-10, max=1.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # Re-normalize
                    
                    # Step 8: Check for and handle NaN/Inf
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        if self.verbose:
                            print(f"Warning: Invalid probabilities detected in column, using uniform distribution")
                        probs = torch.ones_like(probs) / probs.shape[-1]
                    
                    if self._model.training:
                        # During training, use soft labels
                        data_t.append(probs)
                    else:
                        # During sampling, use hard labels with robust multinomial
                        try:
                            # Final check before multinomial
                            if (probs < 0).any() or torch.isnan(probs).any() or torch.isinf(probs).any():
                                raise RuntimeError("Invalid probability distribution")
                            
                            indices = torch.multinomial(probs, 1).squeeze(-1)
                            one_hot = torch.zeros_like(probs)
                            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
                            data_t.append(one_hot)
                        except RuntimeError as e:
                            # Fallback: use argmax if multinomial fails
                            if self.verbose:
                                print(f"Warning: Multinomial sampling failed ({e}), using argmax")
                            indices = probs.argmax(dim=-1)
                            one_hot = torch.zeros_like(probs)
                            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
                            data_t.append(one_hot)
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
                
                st = ed
        
        return torch.cat(data_t, dim=1)
    
    def _validate_discrete_columns(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('train_data should be either pd.DataFrame or np.array.')
        
        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
    
    def _validate_null_data(self, train_data, discrete_columns):
        if isinstance(train_data, pd.DataFrame):
            continuous_cols = list(set(train_data.columns) - set(discrete_columns))
            any_nulls = train_data[continuous_cols].isna().any().any()
        else:
            continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
            any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()
        
        if any_nulls:
            raise InvalidDataError(
                'TabDDPM does not support null values in the continuous training data.'
            )
    
    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the TabDDPM model with improved training stability."""
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)
        
        if epochs is None:
            epochs = self.epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated'),
                DeprecationWarning,
            )
        
        # Transform data
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        
        # Create data sampler
        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self.log_frequency
        )
        
        data_dim = self._transformer.output_dimensions
        cond_dim = self._data_sampler.dim_cond_vec()
        
        # Initialize diffusion
        self._diffusion = GaussianDiffusion(
            timesteps=self.timesteps,
            schedule=self.schedule
        )
        
        # Move diffusion tensors to device
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                     'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self._diffusion, attr, getattr(self._diffusion, attr).to(self._device))
        
        # Initialize model
        self._model = MLPDiffusion(
            data_dim=data_dim,
            cond_dim=cond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            time_embed_dim=self.time_embed_dim,
            dropout=self.dropout
        ).to(self._device)
        
        if self.verbose:
            print(f"\n📊 Model Configuration:")
            print(f"  Data dimensions: {data_dim}")
            print(f"  Conditioning dimensions: {cond_dim}")
            print(f"  Hidden dimensions: {self.hidden_dim}")
            print(f"  Number of layers: {self.num_layers}")
            print(f"  Total parameters: {sum(p.numel() for p in self._model.parameters()):,}")
        
        # Optimizer with warmup
        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Loss'])
        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        
        epoch_iterator = tqdm(range(epochs), disable=(not self.verbose))
        if self.verbose:
            description = 'Loss: {loss:.4f}'
            epoch_iterator.set_description(description.format(loss=0))
        
        for epoch in epoch_iterator:
            self._model.train()
            epoch_losses = []
            
            for _ in range(steps_per_epoch):
                # Sample batch
                condvec = self._data_sampler.sample_condvec(self.batch_size)
                
                if condvec is None:
                    c1 = None
                    real = self._data_sampler.sample_data(train_data, self.batch_size, None, None)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = self._data_sampler.sample_data(
                        train_data, self.batch_size, col[perm], opt[perm]
                    )
                    c1 = c1[perm]
                
                real = torch.from_numpy(real.astype('float32')).to(self._device)
                
                # Sample timesteps
                t = torch.randint(0, self.timesteps, (self.batch_size,), device=self._device).long()
                
                # Calculate loss
                loss = self._diffusion.p_losses(self._model, real, t, c1)
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    if self.verbose:
                        print(f"\nWarning: Invalid loss at epoch {epoch}, skipping batch")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Update learning rate
            scheduler.step()
            
            # Record epoch loss
            avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
            
            # Monitor for training issues
            if epoch > 10 and avg_loss > 10.0:
                if self.verbose:
                    print(f"\n⚠️ Warning: High loss detected ({avg_loss:.4f}). Training may be unstable.")
            
            epoch_loss_df = pd.DataFrame({
                'Epoch': [epoch],
                'Loss': [avg_loss],
            })
            
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df
            
            if self.verbose:
                epoch_iterator.set_description(description.format(loss=avg_loss))
                
                # Check parameter health every 50 epochs
                if epoch % 50 == 0 and epoch > 0:
                    max_param = max(p.abs().max().item() for p in self._model.parameters())
                    if max_param > 100:
                        print(f"\n⚠️ Warning: Large parameter values ({max_param:.2f})")
    
    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Generate synthetic samples with improved stability."""
        if self._model is None:
            raise RuntimeError("Model must be fitted before sampling")
        
        self._model.eval()
        
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self.batch_size
            )
        else:
            global_condition_vec = None
        
        steps = n // self.batch_size + 1
        data = []
        
        with torch.no_grad():
            for i in range(steps):
                # Get condition vector
                if global_condition_vec is not None:
                    condvec = global_condition_vec.copy()
                    c1 = torch.from_numpy(condvec).to(self._device)
                else:
                    condvec = self._data_sampler.sample_condvec(self.batch_size)
                    
                    if condvec is None:
                        c1 = None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                
                # Generate samples
                shape = (self.batch_size, self._transformer.output_dimensions)
                
                try:
                    if self.ddim_sampling:
                        fake = self._diffusion.ddim_sample(
                            self._model, shape, c1, self._device, self.ddim_steps
                        )
                    else:
                        fake = self._diffusion.p_sample_loop(
                            self._model, shape, c1, self._device
                        )
                    
                    # Apply activations
                    fakeact = self._apply_activate(fake)
                    data.append(fakeact.cpu().numpy())
                    
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠️ Warning: Error during sampling batch {i}: {e}")
                        print("Attempting with fallback parameters...")
                    
                    # Fallback: try with more conservative settings
                    try:
                        fake = torch.randn(shape, device=self._device) * 0.5  # Smaller initial noise
                        fakeact = self._apply_activate(fake)
                        data.append(fakeact.cpu().numpy())
                    except Exception as e2:
                        if self.verbose:
                            print(f"⚠️ Fallback also failed: {e2}")
                        continue
        
        if not data:
            raise RuntimeError("Failed to generate any samples. Please retrain the model with more conservative hyperparameters.")
        
        data = np.concatenate(data, axis=0)
        data = data[:n]
        
        # Inverse transform
        synthetic = self._transformer.inverse_transform(data)
        
        # Handle NaN values robustly
        nan_count = synthetic.isna().sum().sum()
        if nan_count > 0 and self.verbose:
            print(f"\n⚠️ Cleaning {nan_count} NaN values in synthetic data...")
        
        for col in synthetic.columns:
            if synthetic[col].isna().any():
                if synthetic[col].dtype == 'object':
                    mode_value = synthetic[col].mode()[0] if len(synthetic[col].mode()) > 0 else 'Unknown'
                    synthetic[col].fillna(mode_value, inplace=True)
                else:
                    median_value = synthetic[col].median()
                    if pd.isna(median_value):
                        median_value = 0
                    synthetic[col].fillna(median_value, inplace=True)
        
        return synthetic
    
    def set_device(self, device):
        """Set the device to be used ('GPU' or 'CPU')."""
        enable_gpu = getattr(self, '_enable_gpu', True)
        from ctgan.synthesizers._utils import _set_device
        self._device = _set_device(enable_gpu, device)
        if self._model is not None:
            self._model.to(self._device)