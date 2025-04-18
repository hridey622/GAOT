import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass, field

from .mlp import ChannelMLP
from .utils.neighbor_search import NeighborSearch
from .geoembed import GeometricEmbedding
from .intergral_transform import IntegralTransform


@dataclass
class MAGNOConfig:
    # GNO parameters
    use_gno: bool = True
    gno_coord_dim: int = 2
    node_embedding: bool = False
    gno_radius: float = 0.033
    gno_use_open3d: bool = False
    gno_use_torch_scatter: str = True
    ## GNOEncoder
    lifting_channels: int = 16
    in_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64, 64])
    in_gno_transform_type: str = 'linear'
    ## GNODecoder
    projection_channels: int = 256
    out_gno_channel_mlp_hidden_layers: list = field(default_factory=lambda: [64, 64])
    out_gno_transform_type: str = 'linear'
    # multiscale aggregation
    scales: list = field(default_factory=lambda: [1.0])
    use_scale_weights: bool = False
    # Attentional aggragation
    use_attn: Optional[bool] = None 
    attention_type: str = 'cosine'
    # Geometric embedding
    use_geoembed: bool = False
    embedding_method: str = 'statistical'
    pooling: str = 'max'
    # Sampling
    sampling_strategy: Optional[str] = None
    max_neighbors: Optional[int] = None     
    sample_ratio: Optional[float] = None 
    # neighbor finding strategy
    neighbor_strategy: str = 'radius'       # ["radius", "knn", "bidirectional"]
    # Dataset
    precompute_edges: bool = True          # Flag for model to load vs compute edges


"""
This code is beta version, the final version will be integrated into the gno.py

I have droped the NeighborSearch_batch and IntegralTransformBatch. Because it is not a big 
for loop, we can use loop to reduce the peak of memory usage.
"""

##########
# MAGNOEncoder
##########
class MAGNOEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        # --- Configuration ---
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edges = gno_config.precompute_edges
        self.use_geoembed = gno_config.use_geoembed
        self.coord_dim = gno_config.gno_coord_dim
        # --- Modules ---
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)

        # Determine GNO input kernel dimension based on config
        in_kernel_in_dim = self.coord_dim * 2
        coord_dim = self.coord_dim 
        if gno_config.node_embedding:
            in_kernel_in_dim = self.coord_dim * 4 * 2 * 2  # 32
            coord_dim = self.coord_dim * 4 * 2
        if gno_config.in_gno_transform_type == "nonlinear" or gno_config.in_gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += in_channels

        # Prepare GNO channel MLP layers
        in_gno_channel_mlp_hidden_layers = gno_config.in_gno_channel_mlp_hidden_layers.copy()
        in_gno_channel_mlp_hidden_layers.insert(0, in_kernel_in_dim)
        in_gno_channel_mlp_hidden_layers.append(out_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.in_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            attention_type=gno_config.attention_type,
            coord_dim=coord_dim,
            sampling_strategy=gno_config.sampling_strategy,
            max_neighbors = gno_config.max_neighbors,
            sample_ratio=gno_config.sample_ratio,
        )

        self.lifting = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=1
        )

        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=out_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1
            )
        
        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.coord_dim = coord_dim
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)

    def forward(self, 
                x_coord: torch.Tensor, 
                pndata: torch.Tensor,
                latent_tokens_coord: torch.Tensor, 
                encoder_nbrs: Optional[list[List[any]]] = None):
        """
        Forward pass for the MAGNO Encoder.

        Args:
            x_coord: Physical node coordinates. Shape: [batch_size, num_nodes, coord_dim]
            pndata: Physical node features. Shape: [batch_size, num_nodes, in_channels]
            latent_tokens_coord: Coordinates of the target latent grid nodes. Shape: [num_latent_nodes, coord_dim]
                                  (Assumed fixed across batch for simplicity here, adjust if varies)
            encoder_nbrs: Optional precomputed neighbor lists for each batch item and scale.
                          Required if self.precompute_edges is True.
                          Format: List[List[neighbors_scale_0, neighbors_scale_1,...]]
                          Outer list length is batch_size, inner list length is num_scales.

        Returns:
            torch.Tensor: Encoded features on the latent grid. Shape: [batch_size, num_latent_nodes, out_channels]
        """

        n_batch, n_nodes, n_coord_dim = x_coord.shape
        n_latent_nodes, _ = latent_tokens_coord.shape

        # 1. Lift input features
        ## Permute for ChannelMLP (1dconv), expect channels first: [batch_size, in_channels, num_nodes]
        pndata = pndata.permute(0,2,1)
        pndata = self.lifting(pndata).permute(0,2,1)
        

        # 2. Prepare scale weights if needed (calculated once based on latent coords)
        # Assuming latent_tokens_coord has shape [m, d] and scale_weighting handles it
        if self.use_scale_weights:
            scale_weights = self.scale_weighting(latent_tokens_coord) # [m, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights) # [m, num_scales]
        
        # 3. Process each item in the batch and Apply GNO and geometric embedding
        batch_encoded_data = []
        for b in range(n_batch):
            x_b = x_coord[b]        # Current batch physical coords [n, d]
            pndata_b = pndata[b]    # Current batch lifted features [n, c_out]
            
            encoded_scales = []
            ## Iterate through each scale
            for scale_idx, scale in enumerate(self.scales):
                ## Determine neighbors for the current scale
                if self.precompute_edges:
                    if encoder_nbrs is None or len(encoder_nbrs) <= b or len(encoder_nbrs[b]) <= scale_idx:
                        raise ValueError(f"Precomputed encoder_nbrs are required but missing or incomplete for batch {b}, scale {scale_idx}.")
                    current_nbrs = encoder_nbrs[b][scale_idx]
                else:
                    ## Recompute neighbors on-the-fly
                    scaled_radius = self.gno_radius * scale
                    with torch.no_grad():
                        current_nbrs = self.nb_search(
                            x_b, latent_tokens_coord, scaled_radius)

                ## Apply GNO transform for this scale
                encoded_unpatched = self.gno(
                    y = x_b,
                    x = latent_tokens_coord,
                    f_y = pndata_b,
                    neighbors = current_nbrs
                ) # shape: [m, c_out]

                ## Apply optional geometric embedding
                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        x_b,
                        latent_tokens_coord,
                        current_nbrs
                    ) # Shape: [m, c_out]

                    ### Combine GNO output and geometric embedding
                    encoded_unpatched = torch.cat([encoded_unpatched, geoembedding], dim=-1) # [m, 2*c_out]
                    encoded_unpatched = encoded_unpatched.unsqueeze(0).permute(0, 2, 1) # [1, 2*c_out, m]
                    encoded_unpatched = self.recovery(encoded_unpatched).permute(0, 2, 1).squeeze(0) # [m, c_out]
                encoded_scales.append(encoded_unpatched)
            
            # Aggregate encoded features across scales
            if len(encoded_scales) == 1:
                encoded_data = encoded_scales[0]        # Shape: [m, c_out]
            else:
                encoded_scales_stack = torch.stack(encoded_scales, dim=0) # # [num_scales, m, c_out]
                if self.use_scale_weights:
                    weights = scale_weights.unsqueeze(-1).permute(1, 0, 2) 
                    encoded_data = (encoded_scales_stack * weights).sum(dim=0)  # [m, c_out]
                else:
                    encoded_data = encoded_scales_stack.sum(dim=0) # [m, c_out]
            batch_encoded_data.append(encoded_data)

        # 4. Stack encoded data across batch
        final_encoded = torch.stack(batch_encoded_data, dim = 0) # Shape: [n_batch, m, c_out]
        return final_encoded


############
# GNO Decoder
############
class MAGNODecoder(nn.Module):
    def __init__(self, in_channels, out_channels, gno_config):
        super().__init__()
        # --- Configuration ---
        self.gno_radius = gno_config.gno_radius
        self.scales = gno_config.scales
        self.use_scale_weights = gno_config.use_scale_weights
        self.precompute_edges = gno_config.precompute_edges
        self.use_geoembed = gno_config.use_geoembed
        self.coord_dim = gno_config.gno_coord_dim

        # --- Modules ---
        self.nb_search = NeighborSearch(gno_config.gno_use_open3d)

        out_kernel_in_dim = self.coord_dim * 2
        if gno_config.node_embedding:
            out_kernel_in_dim = self.coord_dim * 4 * 2 * 2  # 32
            self.coord_dim = self.coord_dim * 4 * 2
        if gno_config.out_gno_transform_type == "nonlinear" or gno_config.out_gno_transform_type == "nonlinear_kernelonly":
            out_kernel_in_dim += out_channels

        # Prepare GNO channel MLP layers
        out_gno_channel_mlp_hidden_layers = gno_config.out_gno_channel_mlp_hidden_layers.copy()
        out_gno_channel_mlp_hidden_layers.insert(0, out_kernel_in_dim)
        out_gno_channel_mlp_hidden_layers.append(in_channels)

        self.gno = IntegralTransform(
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            transform_type=gno_config.out_gno_transform_type,
            use_torch_scatter=gno_config.gno_use_torch_scatter,
            use_attn=gno_config.use_attn,
            attention_type=gno_config.attention_type,
            coord_dim=self.coord_dim,
            sampling_strategy=gno_config.sampling_strategy,
            max_neighbors = gno_config.max_neighbors,
            sample_ratio=gno_config.sample_ratio,
        )

        self.projection = ChannelMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=gno_config.projection_channels,
            n_layers=2,
            n_dim=1,
        )

        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=gno_config.gno_coord_dim,
                output_dim=in_channels,
                method=gno_config.embedding_method,
                pooling=gno_config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )

        if self.use_scale_weights:
            self.num_scales = len(self.scales)
            self.scale_weighting = nn.Sequential(
                nn.Linear(self.coord_dim, 16),
                nn.ReLU(),
                nn.Linear(16, self.num_scales)
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)  

    def forward(self,
                latent_tokens_coord: torch.Tensor, 
                rndata: torch.Tensor,             
                query_coord: torch.Tensor,         
                decoder_nbrs: Optional[List[List[any]]] = None): 
        """
        Forward pass for the MAGNO Decoder.

        Args:
            latent_tokens_coord: Latent grid node coordinates (source). Shape: [num_latent_nodes, coord_dim]
            rndata: Features on the latent grid (source). Shape: [batch_size, num_latent_nodes, in_channels]
            query_coord: Coordinates of the target physical nodes (query points). Shape: [batch_size, num_query_nodes, coord_dim]
            decoder_nbrs: Optional precomputed neighbor lists for each batch item and scale.
                          Required if self.precompute_edges is True.
                          Format: List[List[neighbors_scale_0, neighbors_scale_1,...]]
                          Outer list length is batch_size, inner list length is num_scales.

        Returns:
            torch.Tensor: Decoded features on the physical grid. Shape: [batch_size, num_query_nodes, out_channels]
        """
        n_batch, n_query_nodes, n_coord_dim_query = query_coord.shape
        
        
        # 1. Prepare scale weights if needed (calculated based on query coords)
        if self.use_scale_weights:
            scale_weights = self.scale_weighting(query_coord) # [m, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights) # [m, num_scales]
        
        # 2. Process each item in the batch
        batch_decoded_data = []
        for b in range(n_batch):
            query_coord_b = query_coord[b] # Shape: [m, d]
            rndata_b = rndata[b] # Shape: [n, n_channels]

            decoded_scales = []
            # Iterate through each scale    
            for scale_idx, scale in enumerate(self.scales):
                
                if self.precompute_edges:
                    if decoder_nbrs is None or len(decoder_nbrs) <= b or len(decoder_nbrs[b]) <= scale_idx:
                        raise ValueError(f"Precomputed decoder_nbrs are required but missing or incomplete for batch {b}, scale {scale_idx}.")
                    current_nbrs = decoder_nbrs[b][scale_idx]
                else:
                    ## Recompute neighbors on-the-fly
                    scaled_radius = self.gno_radius * scale
                    with torch.no_grad():
                        current_nbrs = self.nb_search(
                            data = latent_tokens_coord,
                            queries = query_coord_b,
                            radius = scaled_radius
                        )

                decoded_unpatched = self.gno(
                    y = latent_tokens_coord,
                    x = query_coord_b,
                    f_y = rndata_b,
                    neighbors = current_nbrs
                )

                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        input_geom = latent_tokens_coord,
                        latent_queries = query_coord_b,
                        spatial_nbrs = current_nbrs,
                    )

                    combined = torch.cat([decoded_unpatched, geoembedding], dim=-1)
                    combined = combined.unsqueeze(0).permute(0, 2, 1)
                    decoded_unpatched = self.recovery(combined).permute(0, 2, 1).squeeze(0)

                decoded_scales.append(decoded_unpatched)
            
            if len(decoded_scales) == 1:
                decoded_data = decoded_scales[0]
            else:
                decoded_scales_stack = torch.stack(decoded_scales, dim=0)
                if self.use_scale_weights:
                    weights = scale_weights[b].unsqueeze(-1).permute(1, 0, 2)
                    decoded_data = (decoded_scales_stack * weights).sum(dim=0)
                else:
                    decoded_data = decoded_scales_stack.sum(dim=0)
            
            batch_decoded_data.append(decoded_data)

        decoded_combined = torch.stack(batch_decoded_data, dim = 0) # Shape: [n_batch, m, n_channels]

        decoded_combined = decoded_combined.permute(0,2,1)
        projected_decoded = self.projection(decoded_combined)
        final_decoded = projected_decoded.permute(0, 2, 1)

        return final_decoded
