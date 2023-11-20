# Generative Adversarial Network (GAN) for 3D Medical Image Generation

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for generating 3D medical images using the MedNIST dataset.

## Project Structure

The project consists of the following files:

- `generator.py`: Contains the implementation of the generator network, responsible for generating 3D volumes.
- `discriminator.py`: Contains the implementation of the discriminator network, responsible for distinguishing between real and fake 3D volumes.
- `main.py`: The main script that trains the GAN and saves the trained models.
- `README.md`: This file, providing an overview of the project and usage instructions.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib

## Dataset

The MedNIST dataset is used for training the GAN. It contains 6 classes of medical images: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, and HeadCT. Make sure to download and extract the dataset before running the code.

## Usage

1. Install the required dependencies using pip:

pip install torch torchvision matplotlib

2. Download and extract the MedNIST dataset.

3. Open the `main.py` file and modify the `num_epochs`, `batch_size`, `latent_dim`, `channel` and `sample_interval` variables according to your needs.

4. Run the `main.py` script to train the GAN and save the trained models:

5. The trained generator, discriminator, and VAE models will be saved in the current directory as `generator.pt`, `discriminator.pt`, and `vae.pt`, respectively.

6. To load the saved models and generate 3D medical images, you can use the provided code snippet in the `main.py` file after the training loop.

7. The loss values of the discriminator, generator, and VAE during training will be plotted and displayed at the end of the execution.
