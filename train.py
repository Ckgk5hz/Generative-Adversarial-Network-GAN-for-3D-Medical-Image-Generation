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

# Define the path to save the trained models
generator_path = 'generator.pt'
discriminator_path = 'discriminator.pt'
vae_path = 'vae.pt'

# Create empty lists to store loss values
d_loss_values = []
g_loss_values = []
vae_loss_values = []

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(data_loader):
        
        # Configure real images and labels
        real_images = real_images.to(device)
        labels = labels.to(device)

        # Train the discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        real_pred = discriminator(real_images)
        fake_pred = discriminator(fake_images.detach())
        real_loss = adversarial_loss(real_pred, real_labels)
        fake_loss = adversarial_loss(fake_pred, fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        fake_pred = discriminator(fake_images)
        g_loss =adversarial_loss(fake_pred, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Train the VAE
        optimizer_VAE.zero_grad()
        reconstructed_images, mu, logvar = vae(real_images)
        vae_loss = reconstruction_loss(reconstructed_images, real_images) + 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss.backward()
        optimizer_VAE.step()

        # Append loss values to lists
        d_loss_values.append(d_loss.item())
        g_loss_values.append(g_loss.item())
        vae_loss_values.append(vae_loss.item())

        # Print training progress
        if (i+1) % sample_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader)}], "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, VAE_loss: {vae_loss.item():.4f}")

# Save the trained models
torch.save(generator.state_dict(), generator_path)
torch.save(discriminator.state_dict(), discriminator_path)
torch.save(vae.state_dict(), vae_path)

# Load the saved models
generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load(generator_path))
generator.eval()

discriminator = Discriminator(channel).to(device)
discriminator.load_state_dict(torch.load(discriminator_path))
discriminator.eval()

vae = VAE(latent_dim).to(device)
vae.load_state_dict(torch.load(vae_path))
vae.eval()

# Visualize loss
plt.plot(d_loss_values, label='Discriminator Loss')
plt.plot(g_loss_values, label='Generator Loss')
plt.plot(vae_loss_values, label='VAE Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
