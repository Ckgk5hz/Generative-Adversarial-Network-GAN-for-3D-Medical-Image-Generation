import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from generator import SubGenerator, Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 100
batch_size = 64
latent_dim = 1024
channel = 32
num_class = 6
sample_interval = 100
anomaly_threshold = 0.5

# Create the dataset and dataloader
dataset = ImageFolder(root='path/to/mednist', transform=ToTensor())
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Set up the generator, discriminator, and VAE
generator = Generator(latent_dim).to(device)
discriminator = Discriminator(channel).to(device)

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 32 * 32),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(x_hat.size(0), 1, 32, 32, 32)
        return x_hat
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

vae = VAE(latent_dim).to(device)

# Define the loss function and optimizer
adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.MSELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_VAE = optim.Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(data_loader):
        # Adversarial ground truths
        valid = torch.ones(real_images.size(0), 1).to(device)
        fake = torch.zeros(real_images.size(0), 1).to(device)
        
        # Configure input
        real_images = real_images.to(device)
        labels = labels.to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Generate a batch of images
        z = torch.randn(real_images.size(0), latent_dim).to(device)
        gen_images = generator(z)

        # Measure discriminator's ability to classify real and generated samples
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad```python
        # Generate a batch of images
        gen_images = generator(z)

        # Adversarial loss
        g_loss = adversarial_loss(discriminator(gen_images), valid)

        g_loss.backward()
        optimizer_G.step()

        # -----------------
        #  Train VAE
        # -----------------

        optimizer_VAE.zero_grad()

        # Generate a batch of images and reconstruct them
        gen_images = generator(z)
        recon_images, mu, logvar = vae(real_images)

        # Adversarial loss
        vae_loss = adversarial_loss(discriminator(recon_images), valid)

        # Reconstruction loss
        recon_loss = reconstruction_loss(recon_images, real_images)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        vae_loss = vae_loss + recon_loss + kl_loss
        vae_loss.backward()
        optimizer_VAE.step()

        # Print training progress
        if i % sample_interval == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [VAE loss: %f]"
                  % (epoch, num_epochs, i, len(data_loader), d_loss.item(), g_loss.item(), vae_loss.item()))

# Perform anomaly detection
anomaly_scores = []
with torch.no_grad():
    for real_images, labels in data_loader:
        real_images = real_images.to(device)
        recon_images, _, _ = vae(real_images)
        recon_loss = reconstruction_loss(recon_images, real_images)
        anomaly_scores.extend(recon_loss.cpu().numpy())

# Convert 2D images to 3D
def convert_to_3d(images):
    new_images = []
    for image in images:
        new_image = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    if image[i, j, k] > 0.5:
                        new_image.append([i, j, k])
        new_images.append(new_image)
    return new_images

# Convert anomaly scores to z-scores
anomaly_scores = (anomaly_scores - np.mean(anomaly_scores)) / np.std(anomaly_scores)

# Convert real and anomaly images to 3D
real_3d_images = convert_to_3d(real_images.cpu().numpy())
anomaly_3d_images = convert_to_3d(anomaly_images.cpu().numpy())

# Plot the real and anomaly images in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*zip(*real_3d_images), color='b', label='Real')
ax.scatter(*zip(*anomaly_3d_images), color='r', label='Anomaly')
ax.legend()
plt.show()
