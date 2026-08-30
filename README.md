ST1504 CA2 — Deep Learning

Coursework project exploring generative deep learning and reinforcement learning using PyTorch.

The project is divided into two main components:

Part A — Generative Models: Conditional Variational Autoencoders (CVAE) and Conditional GANs for CIFAR-10 image generation.
Part B — Reinforcement Learning: Deep Q-Networks (DQN) for controlling the Pendulum environment under different gravity conditions.
Project Structure
st1504-ca2/
├── notebooks/
│   ├── part_a/
│   │   ├── vae_gan_eda.ipynb
│   │   ├── vae_baseline.ipynb
│   │   ├── vae_improvement.ipynb
│   │   ├── gan_baseline.ipynb
│   │   ├── gan_improvement.ipynb
│   │   └── vae_gan_comparison.ipynb
│   │
│   └── part_b/
│       ├── rl_baseline.ipynb
│       └── rl_improvement.ipynb
│
├── model_checkpoints/
├── models/
├── results/
├── images/
├── papers/
├── contributions/
├── requirements.txt
└── ST1504 CA2.pdf
Part A — Generative Models

Part A investigates class-conditioned image generation on the CIFAR-10 dataset.

Exploratory Data Analysis

vae_gan_eda.ipynb

Shared analysis used by both generative approaches, including:

Class distribution
Sample visualisation
Pixel and colour-channel statistics
Per-class visual complexity
Mean class images
Class similarity
PCA and t-SNE analysis
Colour versus grayscale comparisons
Conditional Variational Autoencoder

vae_baseline.ipynb

Implements a Conditional VAE with:

Conditional encoder and decoder
Reparameterisation trick
Reconstruction and KL-divergence loss
Latent-space visualisation
Latent interpolation
Class-conditioned generation
Generated-image evaluation

vae_improvement.ipynb

Tests multiple improvements to the baseline:

Latent dimensionality
KL weighting / β-VAE
Network depth
Colour vs grayscale inputs
Engineered conditioning features
Data augmentation

The best configuration is subsequently retrained and evaluated as the final VAE.

Conditional GAN

gan_baseline.ipynb

Implements a Conditional DCGAN with:

Conditional generator
Conditional discriminator
Adversarial training
Fixed-noise training progression
Mode-collapse diagnostics
Nearest-neighbour memorisation checks
Class-conditioned image generation

gan_improvement.ipynb

Investigates:

Two-Time-Scale Update Rule (TTUR)
Label smoothing
Spectral normalisation
Colour vs grayscale inputs
Engineered conditioning features
Data augmentation
VAE vs GAN

vae_gan_comparison.ipynb

Provides the final comparison between the two generative architectures based on:

Generated-image quality
Class consistency
Diversity
Training stability
Qualitative sample comparisons
Part B — Reinforcement Learning

Part B investigates whether a Deep Q-Network can learn pendulum control under changes to the environment's gravitational dynamics.

DQN Baseline

rl_baseline.ipynb

The baseline implements:

Experience replay
Target networks
Epsilon-greedy exploration
Action-space discretisation
Neural-network Q-function approximation
Multi-seed evaluation
Training and stability diagnostics

The agent is evaluated under four gravity configurations:

Environment	Gravity
Free fall	g = 0
Anti-gravity	g = -10
Standard gravity	g = 10
Supergravity	g = 15

The baseline DQN outperformed the non-learning reference policies across all tested gravity settings, although performance became less stable under supergravity.

DQN Improvements

rl_improvement.ipynb

The improvement stage evaluates several DQN variants:

Double DQN
Dueling DQN
Double Dueling DQN
Finer torque discretisation

Additional diagnostics include:

Torque saturation
Q-value overestimation
Seed-to-seed variance
Final evaluation return

The strongest configuration is selected using a predefined decision rule before being retrained as the final model.

Tech Stack
Python
PyTorch
Torchvision
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
Hugging Face Datasets
Jupyter
Installation

Clone the repository:

git clone https://github.com/pufferfish3e/st1504-ca2.git
cd st1504-ca2

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate

On Windows:

.venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt

Then launch Jupyter:

jupyter notebook
Recommended Execution Order
Part A
vae_gan_eda.ipynb
        │
        ├──► vae_baseline.ipynb
        │        └──► vae_improvement.ipynb
        │
        └──► gan_baseline.ipynb
                 └──► gan_improvement.ipynb

vae_improvement + gan_improvement
                │
                ▼
      vae_gan_comparison.ipynb
Part B
rl_baseline.ipynb
        │
        ▼
rl_improvement.ipynb
Key Concepts Explored
Generative Deep Learning
Variational Autoencoders
Generative Adversarial Networks
Conditional generation
Latent representations
KL regularisation
Adversarial optimisation
Mode collapse
Image-quality evaluation
Reinforcement Learning
Markov Decision Processes
Q-learning
Deep Q-Networks
Experience replay
Target networks
Double DQN
Dueling architectures
Exploration vs exploitation
Action discretisation
Policy robustness under environmental changes
Repository Notes

Model checkpoints, generated images and experimental result artifacts are retained in the repository to support reproducibility and comparison between experiments.

The project was developed for ST1504 CA2.
