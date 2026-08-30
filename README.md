# ST1504 CA2 — Generative Models and Reinforcement Learning

A PyTorch coursework project investigating two different learning problems: generating class-conditioned images and learning pendulum control under changing gravitational dynamics.

The project follows a **baseline → controlled experiments → final evaluation** workflow, with implementations, visualisations and analysis in Jupyter notebooks.

| Component | Task | Models / environment |
| --- | --- | --- |
| **Part A — Generative modelling** | Generate 32 × 32 RGB images conditioned on CIFAR-10 classes | Conditional VAE and conditional DCGAN |
| **Part B — Reinforcement learning** | Learn pendulum control across four gravity settings | DQN variants in Gymnasium’s `Pendulum-v1` |

## Experiments

### Part A — Conditional image generation

The generative modelling workflow starts with shared CIFAR-10 exploratory analysis, followed by separate VAE and GAN baselines, improvement experiments and a final side-by-side comparison.

The **Conditional Variational Autoencoder (CVAE)** learns a class-conditioned latent representation through reconstruction and KL-divergence losses. The **conditional DCGAN** learns through adversarial training between a generator and discriminator.

| Model | Experiments |
| --- | --- |
| **CVAE** | Latent dimensionality, KL weighting / β-VAE, architecture depth, colour versus grayscale, engineered conditioning features and data augmentation |
| **Conditional GAN** | Two-Time-Scale Update Rule (TTUR), label smoothing, spectral normalisation, colour versus grayscale, engineered conditioning features and data augmentation |

The notebooks also examine latent representations, generated samples, training behaviour and generation quality. The experiments evaluate individual design choices; they are not a claim that every tested technique improves the final model.

### Part B — Pendulum control

The reinforcement learning component implements DQN with experience replay, a target network and epsilon-greedy exploration. Pendulum’s continuous torque range is discretised so the agent can select from a finite set of actions.

The five-action baseline is compared with **Double DQN**, **Dueling DQN**, **Double Dueling DQN** and **vanilla DQN with nine torque bins**. The nine-bin experiment changes the action discretisation without adding the Double or Dueling modifications.

Agents are tested at `g = 10`, `g = 0`, `g = -10` and `g = 15`. The improvement notebook uses three training seeds for the variant experiments and five for the selected final configuration, with a budget of 100,000 environment steps per training run.

## Recorded results

These tables reproduce **saved notebook outputs**, not independently rerun benchmarks.

### Generative model comparison

| Model | NMI ↑ — mean ± standard deviation | FID ↓ |
| --- | ---: | ---: |
| Conditional VAE | 0.1177 ± 0.0103 | 179.6610 |
| Conditional GAN | **0.1982 ± 0.0109** | **122.6940** |

**Evaluation protocol:** each model generates 1,000 images per generation seed, with 100 images for each CIFAR-10 class. NMI is evaluated using the same classifier over generation seeds `123`, `456` and `789`. These are repeated samples from trained models, not three independent training runs. FID uses generation seed `123`, 1,000 generated images and 1,000 real validation images.

NMI measures association between requested classes and classifier-predicted labels; it is not a direct measure of visual realism. FID measures a difference between real and generated feature distributions. The GAN scores better on both metrics in this saved comparison, but the limited FID sample size and classifier-based evaluation constrain the conclusion.

Source: [`vae_gan_comparison.ipynb`](notebooks/part_a/vae_gan_comparison.ipynb). Metrics from separate experiment result files may differ; this table uses the common comparison run only.

### Pendulum control

Higher evaluation return is better.

| Gravity setting | Baseline DQN — mean return | Final nine-action DQN — mean ± standard deviation |
| --- | ---: | ---: |
| Standard — `g = 10` | -133.64 | -135.45 ± 2.05 |
| Zero gravity — `g = 0` | -0.08 | -1964.09 ± 107.94 |
| Anti-gravity — `g = -10` | -109.08 | -108.98 ± 0.62 |
| Supergravity — `g = 15` | -268.80 | -463.94 ± 194.26 |

The final values summarise five training seeds, with 20 final-evaluation episodes per seed.

The nine-action configuration was selected among the tested variants using development evaluations. However, its final results **do not show an overall improvement over the baseline**: standard-gravity and anti-gravity returns remain close to baseline, while zero-gravity and supergravity performance regress substantially.

Source: [`rl_improvement.ipynb`](notebooks/part_b/rl_improvement.ipynb), including its loaded baseline results and final-model evaluation.

## Repository structure

Main notebooks and supporting directories:

```text
st1504-ca2/
├── notebooks/
│   ├── part_a/
│   │   ├── vae_gan_eda.ipynb
│   │   ├── vae_baseline.ipynb
│   │   ├── vae_improvement.ipynb
│   │   ├── gan_baseline.ipynb
│   │   ├── gan_improvement.ipynb
│   │   └── vae_gan_comparison.ipynb
│   └── part_b/
│       ├── rl_baseline.ipynb
│       └── rl_improvement.ipynb
├── models/
├── model_checkpoints/
├── results/
├── images/
├── papers/
├── contributions/
├── requirements.txt
└── ST1504 CA2.pdf
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pufferfish3e/st1504-ca2.git
cd st1504-ca2
python -m venv .venv
```

Use `python3` instead of `python` where required by your installation.

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install the core dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install jupyterlab ipykernel numpy pandas matplotlib seaborn scikit-learn scipy pillow tqdm
python -m pip install datasets torchmetrics torch-fidelity h5py "gymnasium[classic-control]"
```

For a particular CUDA build, replace the PyTorch installation command with the command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your operating system and hardware.

> **Environment note:** `requirements.txt` pins CUDA-specific PyTorch and Torchvision builds (`+cu132`) and omits dependencies used by the notebooks, including Gymnasium and h5py, as well as a Jupyter frontend. It is not a complete, cross-platform installation recipe. The commands above install the core dependencies without those CUDA-specific pins; optional diagnostic cells may require additional packages. This setup has not been independently tested as an exact reproduction of the original training environment.

### 3. Launch JupyterLab

```bash
python -m ipykernel install --user --name st1504-ca2 --display-name "ST1504 CA2"
jupyter lab
```

Select the **ST1504 CA2** kernel when opening a notebook. CIFAR-10 is loaded through Hugging Face Datasets using `uoft-cs/cifar10`; the initial download requires an internet connection.

## Running the notebooks

### Part A

Start with [`vae_gan_eda.ipynb`](notebooks/part_a/vae_gan_eda.ipynb), then run each model’s baseline before its improvement notebook:

```text
vae_gan_eda.ipynb
├── vae_baseline.ipynb → vae_improvement.ipynb
└── gan_baseline.ipynb → gan_improvement.ipynb

Both final models → vae_gan_comparison.ipynb
```

The comparison notebook loads `models/vae_final.pt`, `models/gan_final.pt` and the shared evaluator at `models/vae_classifier.pt`.

### Part B

```text
rl_baseline.ipynb → rl_improvement.ipynb
```

The improvement notebook reads `results/part_b/rl_baseline_results.json`. It defaults to `LOAD_PRETRAINED_MODELS = True`, which loads the expected saved checkpoints and raises an error when a required checkpoint is missing. Set it to `False` to train the configurations instead.

### Paths and saved artifacts

Run notebook kernels with their working directory set to the notebook’s containing folder: `notebooks/part_a/` or `notebooks/part_b/`. Paths such as `../../models` and `../../results` are resolved relative to that directory.

Re-executing notebooks can overwrite result files and checkpoints. Load checkpoints using the matching model definitions and configurations in the notebooks; the presence of a saved checkpoint does not establish that a fresh run will reproduce every reported metric.
