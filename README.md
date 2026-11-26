

Installation

1. Install the following apt packages for mujoco:

   ```bash
   sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
   ```
2. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (strongly recommended) or Anaconda
3. Clone this repo

   ```bash
   git clone https://github.com/＊＊＊/geoflow-policy
   cd geoflow-policy
   ```
4. Install environment:
   Use Mambaforge (strongly recommended):

   ```bash
   mamba env create -f conda_environment.yaml
   conda activate cp
   ```

   or use Anaconda (not recommended):

   ```bash
   conda env create -f conda_environment.yaml
   conda activate cp
   ```
5. Install mimicgen:

   ```bash
   mkdir third_party
   cd third_party
   git clone https://github.com/NVlabs/mimicgen_environments.git
   cd mimicgen_environments
   # This project was developed with Mimicgen v0.1.0. The latest version should work fine, but it is not tested
   git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9
   pip install -e .
   cd ../..
   ```
6. Make sure mujoco version is 2.3.2 (required by mimicgen)

   ```bash
   pip list | grep mujoco
   ```
7. Install additional dependencies:

   ```bash
   pip install torchcfm  # Required for Flow Matching
   ```

## Dataset

### Download Dataset

Download dataset from MimicGen's hugging face: https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core
Make sure the dataset is kept under `/path/to/geoflow-policy/data/robomimic/datasets/[dataset]/[dataset].hdf5`

### Generating Voxel and Point Cloud Observation

```bash
# Template
python scripts/dataset_states_to_obs.py --input data/robomimic/datasets/[dataset]/[dataset].hdf5 --output data/robomimic/datasets/[dataset]/[dataset]_voxel.hdf5 --num_workers=[n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., use 24 workers to generate point cloud and voxel observation for stack_d1 with 200 demos
python scripts/dataset_states_to_obs.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 --num_workers=24 --n=200
```

### Convert Action Space in Dataset

The downloaded dataset has a relative action space. To train with absolute action space, the dataset needs to be converted accordingly

```bash
# Template
python scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/[dataset]/[dataset].hdf5 -o data/robomimic/datasets/[dataset]/[dataset]_abs.hdf5 -n [n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., convert stack_d1_voxel (voxel) with 12 workers
python scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5 -n 12
```

## Training with GeoFlow Policy

To train GeoFlow Policy (with absolute pose control) in Stack D1 task:

```bash
# Make sure you have the voxel converted dataset with absolute action space from the previous step

cd workspace
python train_geoflow_canonical_workspace.py --config-name=train_geoflow_canonical task_name=stack_d1 n_demo=200 

# Adjust GeoFlow parameters
python train_geoflow_canonical_workspace.py --config-name=train_geoflow_canonical task_name=stack_d1 n_demo=200 policy.geoflow_lambda=0.2 policy.use_geoflow=true
```

Alternatively, you can run from the project root:

```bash
python workspace/train_geoflow_canonical_workspace.py --config-name=train_geoflow_canonical task_name=stack_d1 n_demo=200
```

### Key GeoFlow Parameters

- `policy.use_geoflow`: Enable/disable GeoFlow (default: `true`)
- `policy.geoflow_lambda`: Curvature regularization weight (default: `0.001`)
- `policy.geoflow_hidden_dim`: Hidden dimension for GeoFlow coefficient networks (default: `64`)
- `policy.geoflow_delta_scale`: Scale factor for GeoFlow perturbations (default: `0.5`)
- `policy.geoflow_m_grid_points`: Number of grid points for curvature computation (default: `1000`)

## Citation

If you find this work helpful, please cite:

```bash
@article{geoflow-policy,
    title={Geoflow: Geometry-Conditioned Flow Matching With Spatio-Temporal Canonical Representations for Robotic Manipulation},
    year={2026}
}
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement

- This project is built upon [Canonical Policy](https://github.com/ZhangZhiyuanZhang/canonical_policy) and [Equivariant Diffusion Policy](https://github.com/pointW/canonical_policy)
- We used [Vector Neurons](https://github.com/FlyingGiraffe/vnn) for SO(3)-equivariant operations
