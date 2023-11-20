import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import MedicalImageDataset
from generator import Generator
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
num_class = 10
sample_interval = 100
anomaly_threshold = 0.5

# Create the dataset and dataloader
dataset = MedicalImageDataset(...)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Set up the generator and discriminator
generator = Generator(mode="train", latent_dim=latent_dim, channel=channel, num_class=num_class).to(device)
discriminator = Discriminator(channel=channel).to(device)

# Define the loss function and optimizer
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

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
        gen_images = generator(z, class_label=labels)

        # Measure discriminator's ability to classify real and generated samples
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(gen_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_images = generator(z, class_label=labels)

        # Measure generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_images), valid)

        g_loss.backward()
        optimizer_G.step()

        # --------------
        #  Anomaly Detection
        # --------------

        with torch.no_grad():
            # Compute discriminator output on real and generated samples
            real_scores = discriminator(real_images)
            gen_scores = discriminator(gen_images.detach())

            # Compute anomaly score as the absolute difference between real and generated scores
            anomaly_scores = torch.abs(real_scores - gen_scores)

            # Identify anomalies based on anomaly threshold
            anomalies = anomaly_scores > anomaly_threshold

            # Perform further processing or logging with the identified anomalies

        # --------------
        #  2D to 3D Visualization
        # --------------

        if (i + 1) % 10 == 0:
            # Convert a sample of generated 2D images to a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.voxels(gen_images[0].cpu(), facecolors='green', edgecolors='k')
            plt.show()

        # --------------
        #  Log Progress
        # --------------

        if (i + 1) % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch+1, num_epochs, i+1, len(data_loader), d_loss.item(), g_loss.item())
            )

        # Save generated images for every sample_interval iterations
        batches_done = epoch * len(data_loader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
