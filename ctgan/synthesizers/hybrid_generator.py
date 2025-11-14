import torch
from torch import nn
import math

class HybridGenerator(nn.Module):
    """Hybrid Generator combining MLP and lightweight Transformer.
    
    Uses MLP for primary generation and transformer for cross-column dependencies.
    """

    def __init__(self, embedding_dim, generator_dim, data_dim, nhead=2, num_layers=1):
        """Initialize the Hybrid Generator.
        
        Args:
            embedding_dim: Size of the input embedding (noise + conditional vector)
            generator_dim: Original generator dimensions
            data_dim: Output dimension size
            nhead: Number of attention heads for transformer component
            num_layers: Number of transformer layers
        """
        super(HybridGenerator, self).__init__()
        
        # Primary MLP generator (similar to original)
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [
                nn.Linear(dim, item),
                nn.BatchNorm1d(item),
                nn.ReLU()
            ]
            dim = item
        
        # Output layer
        seq.append(nn.Linear(dim, data_dim))
        self.mlp_generator = nn.Sequential(*seq)
        
        # Lightweight transformer for cross-column refinement
        self.hidden_dim = min(128, data_dim)
        self.column_projection = nn.Linear(1, self.hidden_dim)
        
        # Positional embeddings for columns
        self.column_embeddings = nn.Parameter(torch.randn(data_dim, self.hidden_dim))
        
        # Single transformer layer for refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_refiner = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final refinement projection
        self.refinement_projection = nn.Linear(self.hidden_dim, 1)
        
        # Learnable mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.8))  # Start with 80% MLP, 20% transformer

    def forward(self, input_):
        """Apply the Hybrid Generator to the input."""
        batch_size = input_.size(0)
        
        # Primary generation through MLP
        mlp_output = self.mlp_generator(input_)  # [batch_size, data_dim]
        
        # Transformer refinement
        # Project each column value to hidden dimension
        mlp_reshaped = mlp_output.unsqueeze(-1)  # [batch_size, data_dim, 1]
        column_tokens = self.column_projection(mlp_reshaped)  # [batch_size, data_dim, hidden_dim]
        
        # Add positional embeddings
        column_tokens = column_tokens + self.column_embeddings.unsqueeze(0)
        
        # Apply transformer for cross-column refinement
        refined_tokens = self.transformer_refiner(column_tokens)  # [batch_size, data_dim, hidden_dim]
        
        # Project back to single values
        transformer_output = self.refinement_projection(refined_tokens).squeeze(-1)  # [batch_size, data_dim]
        
        # Mix MLP and transformer outputs
        alpha_clamped = torch.sigmoid(self.alpha)  # Ensure alpha is between 0 and 1
        output = alpha_clamped * mlp_output + (1 - alpha_clamped) * transformer_output
        
        return output