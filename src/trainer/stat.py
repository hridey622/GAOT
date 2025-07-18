import os 
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from .base import TrainerBase
from .utils.cal_metric import compute_batch_errors, compute_final_metric
from .utils.plot import plot_estimates
from .utils.data_pairs import CustomDataset

from src.model import init_model

from src.model.layers.utils.neighbor_search import NeighborSearch
from src.utils.scale import rescale, CoordinateScaler

EPSILON = 1e-10
###################
# Utility Functions
#################### 

def move_to_device(data, device):
    """Recursively move all tensors in a nested structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data

def custom_collate_fn(batch):
    """collates data points with potentially variable graph structures."""
    inputs = torch.stack([item[0] for item in batch])          
    labels = torch.stack([item[1] for item in batch])         
    coords = torch.stack([item[2] for item in batch])          
    encoder_graphs = [item[3] for item in batch] 
    decoder_graphs = [item[4] for item in batch]  
    return inputs, labels, coords, encoder_graphs, decoder_graphs


#################
# StaticTrainer_FX Class
#################
class StaticTrainer_FX(TrainerBase):
    """
    Trainer for static problems, and each sample has the same graph structure (coordinates for physical points).
    """
    def __init__(self, args):
        super().__init__(args)

    def _load_and_split_data(self, dataset_config):
        """Loads data, handles specific dataset quirks, splits, and normalizes."""
        # --- Load dataset --- 
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")

        # ----- Sanity-check that the dataset exists before trying to open it -----
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: '{dataset_path}'. "
                "Please verify that 'base_path' and 'name' in your configuration file are correct, "
                "or download / move the required dataset locally."
            )
        self.poseidon_dataset_name = ["Poisson-Gauss", "SE-AF"]
        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values           # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values       # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            if self.metadata.group_x is not None and self.metadata.fix_x == True:
                x_array = ds[self.metadata.group_x].values       # Shape: [num_samples, num_timesteps, num_nodes, num_dims]
                # Some datasets (e.g., Poisson-Gauss) may miss explicit time or channel dimensions.
                # -------------------------------------------------------------
                # 1) Solution / source fields (u_array, c_array)
                if u_array.ndim == 2:
                    # Shape: [num_samples, num_nodes]  → add time & channel dims
                    u_array = u_array[:, np.newaxis, :, np.newaxis]
                elif u_array.ndim == 3:
                    # Shape: [num_samples, num_nodes, num_channels] → add time dim
                    u_array = u_array[:, np.newaxis, :, :]

                if c_array is not None:
                    if c_array.ndim == 2:
                        c_array = c_array[:, np.newaxis, :, np.newaxis]
                    elif c_array.ndim == 3:
                        c_array = c_array[:, np.newaxis, :, :]

                # -------------------------------------------------------------
                # 2) Coordinates (x_array)
                # Possible shapes:
                #   [num_nodes]                       → 1-D, need to expand to 2-D coordinate (x,0)
                #   [num_nodes, 2]                   → shared across samples
                #   [num_samples, num_nodes, num_dims]→ per-sample coords
                if x_array.ndim == 1:
                    # Treat as 1-D coordinate along x; create y=0 dummy column.
                    x_array = np.stack((x_array, np.zeros_like(x_array)), axis=-1)  # [num_nodes, 2]
                    x_array = x_array[np.newaxis, np.newaxis, :, :]               # [1,1,num_nodes,2]
                elif x_array.ndim == 2:
                    # Shared coords [num_nodes, num_dims]
                    x_array = x_array[np.newaxis, np.newaxis, :, :]
                elif x_array.ndim == 3:
                    # Per-sample coords [num_samples, num_nodes, num_dims]
                    x_array = x_array[:, np.newaxis, :, :]
            elif self.metadata.group_x is not None and self.metadata.fix_x == False:
                raise ValueError("fix_x must be True for StaticTrainer_FX. Otherwise change to use StaticTrainer_VX.")
            else:
                domain_x = self.metadata.domain_x               # Shape: ([x_min, y_min], [x_max, y_max])
                nx, ny = u_array.shape[-2], u_array.shape[-1]
                x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
                y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
                x_array = np.stack((xv, yv), axis=-1)               # Shape: [num_nodes, 2]
                x_array = x_array.reshape(-1, 2)                    # Shape: [num_nodes, 2]
                num_nodes = x_array.shape[0]
                num_samples = u_array.shape[0]
                
                c_array = c_array.reshape(num_samples,-1, c_array.shape[-1])if c_array is not None else None
                u_array = u_array.reshape(num_samples,-1, u_array.shape[-1]) # Shape: [num_samples, num_nodes, num_channels]
                node_permutation = np.random.permutation(num_nodes)
                x_array = x_array[node_permutation, :]
                u_array = u_array[:, node_permutation, :]
                if c_array is not None:
                    c_array = c_array[:, node_permutation, :]
                x_array = x_array[np.newaxis, np.newaxis, :, :]             # Shape: [1, 1, num_nodes, 2]
                u_array = u_array[:, np.newaxis, :, :]                     # Shape: [num_samples, 1, num_nodes, num_channels]
                if c_array is not None:
                    c_array = c_array[:, np.newaxis, :, :]                 # Shape: [num_samples, 1, num_nodes, num_channels_c]

        # -------- Final sanity check on dimensions --------
        # Ensure tensors have 4D shapes: [num_samples, 1, num_nodes, channels]
        if u_array.ndim == 2:
            u_array = u_array[:, np.newaxis, :, np.newaxis]
        elif u_array.ndim == 3:
            u_array = u_array[:, np.newaxis, :, :]
        if c_array is not None:
            if c_array.ndim == 2:
                c_array = c_array[:, np.newaxis, :, np.newaxis]
            elif c_array.ndim == 3:
                c_array = c_array[:, np.newaxis, :, :]
        # Ensure x_array has shape [*,1,num_nodes,2]
        if x_array.ndim == 2:
            x_array = x_array[np.newaxis, np.newaxis, :, :]
        elif x_array.ndim == 3:
            x_array = x_array[:, np.newaxis, :, :]

        # --- Dataset Specific Handling ---
        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            x_array = x_array[:,:,:9216,:]
        
        # --- Select Variables & Check Shapes ---
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1] if c_array is not None else 0
        self.num_output_channels = u_array.shape[-1]

        # --- Compute Sizes & Indices ---
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."

        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])
        ## x_array is not split, it is the same for all samples, x_train = x_array[0, 0]  # Shape: [num_nodes, 2]
        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None
        
        # --- Compute Statics & Normalize (using training set only) ---
        print("Computing statistics and normalizing data")
        print("You need to make sure that the u_mean and u_std are the same for testing other datasets!")
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std
        self.c_mean = None
        self.c_std = None
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std
            c_train = torch.tensor(c_train, dtype=self.dtype).squeeze(1)
            c_val = torch.tensor(c_val, dtype=self.dtype).squeeze(1)
            c_test = torch.tensor(c_test, dtype=self.dtype).squeeze(1)
        
        # --- Convert to Tensors ---
        # Handle None case for c_train/val/test when converting
        u_train = torch.tensor(u_train, dtype=self.dtype).squeeze(1)
        u_val = torch.tensor(u_val, dtype=self.dtype).squeeze(1)
        u_test = torch.tensor(u_test, dtype=self.dtype).squeeze(1)
        x_tensor = torch.tensor(x_array, dtype=self.dtype)
        # Robustly extract the coordinate matrix (shape: [num_nodes, 2]) regardless of how many singleton
        # dimensions remain. After the earlier squeeze we may still have 2‒4 dims.
        x_tensor_squeezed = x_tensor.squeeze()  # remove all size-1 axes

        if x_tensor_squeezed.ndim == 2:              # [num_nodes, 2]
            x_train = x_tensor_squeezed
        elif x_tensor_squeezed.ndim == 3:            # [num_samples, num_nodes, 2]
            x_train = x_tensor_squeezed[0]
        elif x_tensor_squeezed.ndim == 4:            # [num_samples, num_timesteps, num_nodes, 2]
            x_train = x_tensor_squeezed[0, 0]
        else:
            raise ValueError(f"Unexpected coordinate tensor shape: {x_tensor.shape}")
        # For FX problems coordinates are identical across samples, so reuse x_train for val/test.
        x_val = x_train
        x_test = x_train

        return {
            "train": {"c": c_train, "u": u_train, "x": x_train},
            "val":   {"c": c_val,   "u": u_val,   "x": x_val},
            "test":  {"c": c_test,  "u": u_test,  "x": x_test},
        }
    
    def _generate_latent_queries(self, token_size = (64, 64), coord_scaler = None):
        """Generates latent query points on a regular grid."""
        phy_domain = self.metadata.domain_x
        x_min, y_min = phy_domain[0]
        x_max, y_max = phy_domain[1]

        if not isinstance(token_size[0], int) or not isinstance(token_size[1], int):
            raise ValueError("token_size must be a tuple of two integers.")

        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, token_size[0], dtype=self.dtype), 
            torch.linspace(y_min, y_max, token_size[1], dtype=self.dtype), 
            indexing='ij' 
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1,2)
        latent_queries = coord_scaler(latent_queries)

        return latent_queries

    def init_dataset(self, dataset_config):
        # --- 1. Load, Split, Normalize Data ---
        print("Loading and preprocessing data...")
        data_splits = self._load_and_split_data(dataset_config)
        c_train, u_train, x_train = data_splits["train"]["c"], data_splits["train"]["u"], data_splits["train"]["x"]
        c_val,   u_val,   x_val   = data_splits["val"]["c"],   data_splits["val"]["u"],   data_splits["val"]["x"]
        c_test,  u_test,  x_test  = data_splits["test"]["c"],  data_splits["test"]["u"],  data_splits["test"]["x"]

        # --- 2. Prepare for Latent Tokens --- 
        coord_scaler = CoordinateScaler(
            target_range=(-1, 1),
            mode = dataset_config.coord_scaling)
        latent_queries = self._generate_latent_queries(
            token_size = self.model_config.latent_tokens_size,
            coord_scaler = coord_scaler
        )
        self.latent_tokens_coord = latent_queries
        self.coord = coord_scaler(x_train)

        # --- 3. Build Graphs ---
        train_ds = TensorDataset(c_train, u_train)
        val_ds  = TensorDataset(c_val,   u_val)
        test_ds = TensorDataset(c_test,  u_test)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=dataset_config.batch_size,
            shuffle=dataset_config.shuffle,
            num_workers=dataset_config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=dataset_config.num_workers,
            pin_memory=True
        )
        print("Data loading and preprocessing complete.")
    
    def init_model(self, model_config):
        self.model = init_model(
                input_size = self.num_input_channels,
                output_size = self.num_output_channels,
                model = model_config.name,
                config = model_config
            )
    
    def train_step(self, batch):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)
        pred = self.model(
            latent_tokens_coord = latent_tokens_coord,
            xcoord = coord, 
            pndata = x_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                coord = self.coord.to(self.device)
                pred = self.model(
                    latent_tokens_coord = latent_tokens_coord,
                    xcoord = coord, 
                    pndata = x_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample) in enumerate(self.test_loader):
                x_sample, y_sample = x_sample.to(self.device), y_sample.to(self.device)
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                coord = self.coord.to(self.device)
                pred = self.model(
                    latent_tokens_coord = latent_tokens_coord, 
                    xcoord = coord, 
                    pndata = x_sample)
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_sample * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric
        print(f"relative error: {final_metric}")
        x_sample_de_norm = x_sample * self.c_std.to(self.device) + self.c_mean.to(self.device)

        fig = plot_estimates(
            u_inp = x_sample_de_norm[-1].cpu().numpy(), 
            u_gtr = y_sample_de_norm[-1].cpu().numpy(), 
            u_prd = pred_de_norm[-1].cpu().numpy(), 
            x_inp = coord.cpu().numpy(),
            x_out = coord.cpu().numpy(),
            names = self.metadata.names['c'],
            symmetric = self.metadata.signed['u'])
        
        fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Plot saved to {self.path_config.result_path}")



##################
# StaticTrainer_VX Class
#################
class StaticTrainer_VX(TrainerBase):
    """
    Trainer for static problems, and each sample has different graph structures (coordinates for physical points).
    """
    def __init__(self, args):
        super().__init__(args)
   
    def _load_and_split_data(self, dataset_config):
        """Loads data, handles specific dataset quirks, splits, and normalizes."""
        # --- Load dataset --- 
        base_path = dataset_config.base_path
        dataset_name = dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")

        # ----- Sanity-check that the dataset exists before trying to open it -----
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: '{dataset_path}'. "
                "Please verify that 'base_path' and 'name' in your configuration file are correct, "
                "or download / move the required dataset locally."
            )
        self.poseidon_dataset_name = ["Poisson-Gauss"]
        with xr.open_dataset(dataset_path) as ds:
            u_array = ds[self.metadata.group_u].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values  # Shape: [num_samples, num_timesteps, num_nodes, num_channels_c]
            else:
                c_array = None
            # Load x
            if self.metadata.group_x is not None and self.metadata.fix_x == False:
                x_array = ds[self.metadata.group_x].values
                if x_array.shape[0] == u_array.shape[0]:
                   x_array = x_array
                   self.x_train = x_array    # [num_samples, num_timesteps, num_nodes, num_dims]
            else:
                raise ValueError("fix_x must be False for unstructured data")
        
        # --- Dataset Specific Handling ---
        if dataset_name in self.poseidon_dataset_name and dataset_config.use_sparse:
            u_array = u_array[:,:,:9216,:]
            if c_array is not None:
                c_array = c_array[:,:,:9216,:]
            self.x_train = self.x_train[:,:,:9216,:]
        
        # --- Select Variables & Check Shapes ---
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        self.num_input_channels = c_array.shape[-1]
        self.num_output_channels = u_array.shape[-1]

        # --- Compute Sizes & Indices ---
        total_samples = u_array.shape[0]
        train_size = dataset_config.train_size
        val_size = dataset_config.val_size
        test_size = dataset_config.test_size
        assert train_size + val_size + test_size <= total_samples, "Sum of train, val, and test sizes exceeds total samples"
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets."
    
        if dataset_config.rand_dataset:
            indices = np.random.permutation(len(u_array))
        else:
            indices = np.arange(len(u_array))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[-test_size:]

        # Split data into train, val, test
        u_train = u_array[train_indices]
        u_val = u_array[val_indices]
        u_test = u_array[test_indices]
        x_train = self.x_train[train_indices]
        x_val = self.x_train[val_indices]
        x_test = self.x_train[test_indices]
        if c_array is not None:
            c_train = c_array[train_indices]
            c_val = c_array[val_indices]
            c_test = c_array[test_indices]
        else:
            c_train = c_val = c_test = None

        # --- Compute Statics & Normalize (using training set only) ---
        print("Computing statistics and normalizing data")
        print("You need to make sure that the u_mean and u_std are the same for testing other datasets!")
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON  # Avoid division by zero
        # u_mean = np.array(0.9530)
        # u_std = np.array(0.3153)
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)
        u_train = (u_train - u_mean) / u_std
        u_val = (u_val - u_mean) / u_std
        u_test = (u_test - u_mean) / u_std

        self.c_mean = None
        self.c_std = None
        if c_array is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON
            # c_mean = np.array([0.8046, 7.6054, 2.0414])
            # c_std = np.array([0.3062, 4.3355, 2.4575])
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)
            c_train = (c_train - c_mean) / c_std
            c_val = (c_val - c_mean) / c_std
            c_test = (c_test - c_mean) / c_std
            c_train = torch.tensor(c_train, dtype=self.dtype)
            c_val = torch.tensor(c_val, dtype=self.dtype)
            c_test = torch.tensor(c_test, dtype=self.dtype)
        # --- Convert to Tensors ---
        # Handle None case for c_train/val/test when converting
        u_train = torch.tensor(u_train, dtype=self.dtype)
        u_val = torch.tensor(u_val, dtype=self.dtype)
        u_test = torch.tensor(u_test, dtype=self.dtype)
        x_train = torch.tensor(x_train, dtype=self.dtype)
        x_val = torch.tensor(x_val, dtype=self.dtype)
        x_test = torch.tensor(x_test, dtype=self.dtype)

        # Return dictionary for clarity
        return {
            "train": {"c": c_train, "u": u_train, "x": x_train},
            "val":   {"c": c_val,   "u": u_val,   "x": x_val},
            "test":  {"c": c_test,  "u": u_test,  "x": x_test},
        }

    def _generate_latent_queries(self, token_size = (64, 64), coord_scaler = None):
        """Generates latent query points on a regular grid."""
        phy_domain = self.metadata.domain_x
        x_min, y_min = phy_domain[0]
        x_max, y_max = phy_domain[1]

        if not isinstance(token_size[0], int) or not isinstance(token_size[1], int):
            raise ValueError("token_size must be a tuple of two integers.")

        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, token_size[0], dtype=self.dtype), 
            torch.linspace(y_min, y_max, token_size[1], dtype=self.dtype), 
            indexing='ij' 
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1,2)
        latent_queries = coord_scaler(latent_queries)

        return latent_queries

    def _build_graphs_for_split(self, x_data, latent_queries, nb_search, gno_radius, scales):
        """Builds encoder and decoder graphs for a given data split (train/val/test)."""
        encoder_graphs_split = []
        decoder_graphs_split = []
        num_samples = x_data.shape[0]

        for i in range(num_samples):
            if x_data.dim() == 4 and x_data.shape[1] == 1:
                x_coord = rescale(x_data[i, 0], (-1, 1))
            elif x_data.dim() == 3:
                x_coord = rescale(x_data[i], (-1, 1))
            else:
                raise ValueError(f"Unexpected shape for x_data: {x_data.shape}")
        
            encoder_nbrs_sample = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = nb_search(x_coord, latent_queries, scaled_radius)
                encoder_nbrs_sample.append(nbrs)
            encoder_graphs_split.append(encoder_nbrs_sample)

            decoder_nbrs_sample = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = nb_search(latent_queries, x_coord, scaled_radius)
                decoder_nbrs_sample.append(nbrs)
            decoder_graphs_split.append(decoder_nbrs_sample)

        return encoder_graphs_split, decoder_graphs_split

    def init_dataset(self, dataset_config):
        # --- 1. Load, Split, Normalize Data ---
        print("Loading and preprocessing data...")
        data_splits = self._load_and_split_data(dataset_config)
        # Extract tensors for convenience
        c_train, u_train, x_train = data_splits["train"]["c"], data_splits["train"]["u"], data_splits["train"]["x"]
        c_val,   u_val,   x_val   = data_splits["val"]["c"],   data_splits["val"]["u"],   data_splits["val"]["x"]
        c_test,  u_test,  x_test  = data_splits["test"]["c"],  data_splits["test"]["u"],  data_splits["test"]["x"]


        # --- 2. Prepare for Graph Building --- 
        print("Preparing for graph building...")
        nb_search = NeighborSearch(self.model_config.args.magno.gno_use_open3d)
        gno_radius = self.model_config.args.magno.gno_radius
        scales = self.model_config.args.magno.scales
        coord_scaler = CoordinateScaler(
            target_range=(-1, 1),
            mode = dataset_config.coord_scaling)
        latent_queries = self._generate_latent_queries(
            token_size = self.model_config.latent_tokens_size,
            coord_scaler = coord_scaler
        )
        self.latent_tokens_coord = latent_queries

        # --- 3. Build Graphs ---
        print("Starting Graph Build ...")
        graph_start_time = time.time()
        
        encoder_graphs_test, decoder_graphs_test = self._build_graphs_for_split(
            x_test, latent_queries, nb_search, gno_radius, scales
        )

        if self.setup_config.train: 
            encoder_graphs_train, decoder_graphs_train = self._build_graphs_for_split(
                x_train, latent_queries, nb_search, gno_radius, scales
            )
            print(f"Built Train Graphs ({len(x_train)} samples)...")

            encoder_graphs_val, decoder_graphs_val = self._build_graphs_for_split(
                x_val, latent_queries, nb_search, gno_radius, scales
            )
            print(f"Built Val Graphs ({len(x_val)} samples)...")
            

            train_ds = CustomDataset(
                c_train, u_train, x_train, encoder_graphs_train, decoder_graphs_train,
                transform=coord_scaler)
            self.train_loader = DataLoader(
                train_ds, 
                batch_size=dataset_config.batch_size, 
                shuffle=dataset_config.shuffle, 
                collate_fn=custom_collate_fn,
                num_workers=dataset_config.num_workers,
                pin_memory=True
                )
            val_ds   = CustomDataset(
                c_val,   u_val,   x_val,   encoder_graphs_val,   decoder_graphs_val,
                transform=coord_scaler)
            self.val_loader = DataLoader(
                val_ds, 
                batch_size=dataset_config.batch_size, 
                shuffle=False, 
                collate_fn=custom_collate_fn,
                num_workers=dataset_config.num_workers,
                pin_memory=True
                )
        else:
            self.train_loader = None
            self.val_loader = None
            print("Skipping Train/Validation graph build as setup_config.train is False.")
        
        print(f"Graph building took {time.time() - graph_start_time:.2f} s!")
        
        
        test_ds  = CustomDataset(
            c_test,  u_test,  x_test,  encoder_graphs_test,  decoder_graphs_test,
            transform=coord_scaler)

        self.test_loader = DataLoader(
            test_ds, 
            batch_size=dataset_config.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=dataset_config.num_workers,
            pin_memory=True
            )
                                    
    def init_model(self, model_config):
        self.model = init_model(
                input_size = self.num_input_channels,
                output_size = self.num_output_channels,
                model = model_config.name,
                config = model_config
            )

    def train_step(self, batch):
        x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
        x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
        encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
        decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        pred = self.model(
            latent_tokens_coord = latent_tokens_coord,
            xcoord = coord_batch, 
            pndata = x_batch,
            encoder_nbrs = encoder_graph_batch, 
            decoder_nbrs = decoder_graph_batch)
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch in loader:
                x_batch, y_batch, coord_batch = x_batch.to(self.device), y_batch.to(self.device), coord_batch.to(self.device)
                encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
                decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                pred = self.model(
                    latent_tokens_coord = latent_tokens_coord,
                    xcoord = coord_batch, 
                    pndata = x_batch, 
                    encoder_nbrs = encoder_graph_batch, 
                    decoder_nbrs = decoder_graph_batch)
                loss = self.loss_fn(pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test(self):
        self.model.eval()
        self.model.to(self.device)
        all_relative_errors = []
        with torch.no_grad():
            for i, (x_sample, y_sample, coord_sample, encoder_graph_sample, decoder_graph_sample) in enumerate(self.test_loader):
                x_sample, y_sample, coord_sample = x_sample.to(self.device), y_sample.to(self.device), coord_sample.to(self.device) # Shape: [batch_size, num_timesteps, num_nodes, num_channels]
                encoder_graph_sample = move_to_device(encoder_graph_sample, self.device)
                decoder_graph_sample = move_to_device(decoder_graph_sample, self.device)
                latent_tokens_coord = self.latent_tokens_coord.to(self.device)
                pred = self.model(
                    latent_tokens_coord = latent_tokens_coord, 
                    xcoord = coord_sample, 
                    pndata = x_sample, 
                    encoder_nbrs = encoder_graph_sample, 
                    decoder_nbrs = decoder_graph_sample)
                
                pred_de_norm = pred * self.u_std.to(self.device) + self.u_mean.to(self.device)
                y_sample_de_norm = y_sample * self.u_std.to(self.device) + self.u_mean.to(self.device)
                relative_errors = compute_batch_errors(y_sample_de_norm, pred_de_norm, self.metadata)
                all_relative_errors.append(relative_errors)
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric
        print(f"relative error: {final_metric}")

        x_sample_de_norm = x_sample * self.c_std.to(self.device) + self.c_mean.to(self.device)
   
        fig = plot_estimates(
            u_inp = x_sample_de_norm[-1].cpu().numpy(), 
            u_gtr = y_sample_de_norm[-1].cpu().numpy(), 
            u_prd = pred_de_norm[-1].cpu().numpy(), 
            x_inp = coord_sample[-1].cpu().numpy(),
            x_out = coord_sample[-1].cpu().numpy(),
            names = self.metadata.names['c'],
            symmetric = self.metadata.signed['u'])

        fig.savefig(self.path_config.result_path,dpi=300,bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

