# :goat: GAOT: Geometry Aware Operator Transformer

This is the source code for the paper:  
**"Geometry Aware Operator Transformer As An Efficient And Accurate Neural Surrogate For PDEs On Arbitrary Domains"**

## Abstract

The very challenging task of learning solution operators of PDEs on arbitrary domains accurately and efficiently is of vital importance to engineering and industrial simulations. Despite the existence of many operator learning algorithms to approximate such PDEs, we find that accurate models are not necessarily computationally efficient and vice versa. We address this issue by proposing a geometry aware operator transformer (GAOT) for learning PDEs on arbitrary domains. GAOT combines novel multiscale attentional graph neural operator encoders and decoders, together with geometry embeddings and (vision) transformer processors to accurately map information about the domain and the inputs into a robust approximation of the PDE solution. Multiple innovations in the implementation of GAOT also ensure computational efficiency and scalability. We demonstrate this significant gain in both accuracy and efficiency of GAOT over several baselines on a large number of learning tasks from a diverse set of PDEs, including achieving state of the art performance on a large scale three-dimensional industrial CFD dataset.

<p align="center">
  <img src="assets/architecture.png" alt="architecture" width="900"/>
</p>

## Results

### Overall Model Performance

The GAOT model exhibits superior performance across multiple metrics when compared to the selected baselines (RIGNO-18 for Graph-based, GINO for FNO-based, and Transolver for Transformer-based models). The radar chart below provides an overview of GAOT's performance characteristics.

<p align="center">
  <img src="assets/gaot_model_performance_radar_proportional.png" alt="GAOT Model Performance Radar Chart" width="600"/>
  <br/>
  <em>Figure 1: Normalized performance of GAOT and baselines across eight axes, covering accuracy (Acc.), robustness (Robust), throughput (Tput), scalability (Scal.) on time-dependent (TD) and time-independent (TI) tasks.</em>
</p>

### 📈 Throughput and Scalability

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="assets/grid_vs_throughput.png" alt="Grid Resolution vs. Throughput" width="90%"/><br/>
      <em>Figure 2: Grid vs. Throughput</em>
    </td>
    <td align="center" width="50%">
      <img src="assets/model_vs_throughput.png" alt="Model vs. Throughput" width="90%"/><br/>
      <em>Figure 3: Model vs. Throughput</em>
    </td>
  </tr>
</table>



## Installation

1. **Create and activate a virtual environment:**

   ```bash
   python -m virtualenv venv-gaot
   source venv-gaot/bin/activate
   ```

2. **Install the necessary packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

Download and place your datasets in a directory structure similar to the following:

``` 
.../your_base_dataset_directory/
    |__ time_indep/
        |__ Poisson-Gauss.nc
        |__ naca0012.nc
        |__ ...
    |__ time_dep/
        |__ ns_gauss.nc
        |__ ...
```

You will specify `your_base_dataset_directory/` in the configuration files (see `dataset.base_path` below).

## How to use

### Configuration

All experiment parameters are defined in configuration files (JSON or TOML format) located in the `config/` directory.

Key parameters within the configuration files include: 

-  `dataset.base_path`: Should be set to the path of `your_base_dataset_directory/` mentioned above. 
-  `dataset.name`: Should correspond to the name of your dataset file (e.g., "Poisson-Gauss" for "Poisson-Gauss.nc").
- `setup.train`: Set to `true` for training, `false` for inference. 
- `setup.test`: Set to `true` for testing/inference (usually after setting `setup.train: false`). 
- `path`: Defines where checkpoints, loss plots, result visualizations, and the metrics database will be stored (e.g., in `.ckpt/`, `.loss/`, `.result/`, `.database/,` respectively).

For a detailed explanation of all available configuration options and their default values, please refer to: 

```
.../src/
    |__ trainer/
        |__ utils/
          |__ default_set.py
```

#### Model and Trainer Selection
This repository supports different GAOT models and trainer types to cater to different problem setups. These are specified in your configuration file:
- Trainer Selection (`setup.trainer_name`):
  - `static_fx`: For time-independent datasets where the geometry (coordinates) is fixed (identical) across all data samples.
  - `static_vx`: For time-independent datasets where the geometry (coordinates) is variable (differs) across data samples.
  - `sequential_fx`: For time-dependent datasets where the geometry (coordinates) is fixed across all data samples and time steps.

- Model Selection (`model.name`):
  - `goat2d_fx`: A 2D GOAT model designed for datasets with fixed geometry.
  - `goat2d_vx`: A 2D GOAT model designed for datasets with variable geometry.

Carefully choose the `trainer_name` and `model.name` in your configuration file to match the characteristics of your dataset and the problem you are solving. The default settings for these can be found in `src/trainer/utils/default_set.py`.



Example configuration files are provided in the `config/` directory:

```
.../config/examples/
            |__ time_indep/
                |__ poisson_gauss.json
                |__ naca0012.json
            |__ time_dep/
                |__ ns_gauss.json
```

### Training

To train a model, run `main.py` with the path to your desired configuration file:

```bash
python main.py --config [path_to_your_config_file.json_or_toml]
```

For example:

```bash
python main.py --config config/static/poisson_gauss.json
```

You can also run all configuration files within a specific folder:

```bash
python main.py --folder [path_to_your_config_folder]
```

Other command-line options for `main.py` include:

- `--debug`: Run in debug mode (may affect multiprocessing).
- `--num_works_per_device <int>`: Number of parallel workers per device.
- `--visible_devices <int ...>`: Specify which CUDA devices to use (e.g., `--visible_devices 0 1`).

During training, checkpoints, loss curves, visualizations, and a CSV database of metrics will be saved to the directories specified in the `path` section of your configuration file.

### Inference

To run inference using a trained model:

1. Modify your configuration file:

   - Set `setup.train: false`.
   - Set `setup.test: true`.
   - Ensure `path.ckpt_path` in the config points to the desired model checkpoint file.

2. Run `main.py` with the updated configuration file:

   ```
   python main.py --config [path_to_your_config_file.json_or_toml]
   ```

### Project Structure Overview
```
GOAT/
├── config/                   # Experiment configuration files (.json, .toml)
├── demo/                     # Jupyter notebooks, scripts for demos, analysis
├── src/                      # Source code
│   ├── data/                 # Data loading (dataset.py)
│   ├── model/                # Model definitions (goat2d_fx.py, goat2d_vx.py, layers/)
│   ├── trainer/              # Training, evaluation, optimizers, utils (default_set.py)
│   └── utils/                # General utility functions
├── assets/                   # (Saved images like architecture.png)
├── main.py                   # Main script for running experiments
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Citation
```
@article{wen2025goat,
  title={Geometry Aware Operator Transformer As An Efficient And Accurate Neural Surrogate For PDEs On Arbitrary Domains},
  author={Wen, Shizheng and Kumbhat, Arsh and Lingsch, Levi and Mousavi, Sepehr and Chandrashekar, Praveen and Mishra, Siddhartha},
  journal={arXiv preprint arXiv:xxxxx},
  year={2025}
}
```