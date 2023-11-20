import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from pyod.models.auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generator import Generator

# Rule 1: GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        return x

# Rule 2: 2D to 3D Visualization
def visualize_2d_to_3d(original_2d, generated_3d):
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.imshow(original_2d, cmap='gray')
    ax1.set_title('Original 2D Image')

    ax2 = fig.add_subplot(122, projection='3d')
    x, y = torch.meshgrid(torch.linspace(0, 1, generated_3d.shape[0]), torch.linspace(0, 1, generated_3d.shape[1]))
    ax2.plot_surface(x, y, generated_3d, cmap='gray')
    ax2.set_title('Generated 3D Image')

    plt.show()

# Rule 3: Anomaly Detection for Medical Images
class MedNISTDataset:
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
        self.classes = self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label

# Hyperparameters
latent_dim = 100
batch_size = 32
num_epochs = 10
data_dir = 'path_to_dataset_directory'

# Data Preparation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MedNISTDataset(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# GAN Models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Anomaly Detection Model
model = AutoEncoder(hidden_neurons=[256, 128, 32, 128, 256], epochs=10)

# Loss Function
criterion = nn.BCELoss()

# Optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
model_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)
model.to(device)

# Training Loop
for epoch in range(num_epochs):
    generator_loss_total = 0.0
    discriminator_loss_total = 0.0
    anomaly_loss_total = 0.0

    for i, (images, _) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.to(device)

        # Rule 1: GAN Training
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        real_outputs = discriminator(images)
        real_loss = criterion(real_outputs, real_labels)

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(noise)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        fake_outputs = discriminator(fake_images)
        generator_loss = criterion(fake_outputs, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

        generator_loss_total += generator_loss.item()
        discriminator_loss_total += discriminator_loss.item()

        # Rule 3: Anomaly Detection
        model_optimizer.zero_grad()
        generated_3d = model.predict(torch.flatten(fake_images, start_dim=1))
        original_2d = torch.squeeze(images, dim=1)
        anomaly_loss = criterion(generated_3d, original_2d)

        anomaly_loss.backward()
        model_optimizer.step()

        anomaly_loss_total += anomaly_loss.item()

    # Print epoch statistics
    generator_loss_avg = generator_loss_total / len(train_loader)
    discriminator_loss_avg = discriminator_loss_total / len(train_loader)
    anomaly_loss_avg = anomaly_loss_total / len(train_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Generator Loss: {generator_loss_avg:.4f}, '
          f'Discriminator Loss: {discriminator_loss_avg:.4f}, '
          f'Anomaly Loss: {anomaly_loss_avg:.4f}')

    # Visualize generated 3D image for the first batch of the last epoch
    if epoch == num_epochs - 1:
        visualize_2d_to_3d(original_2d[0].cpu(), generated_3d[0].cpu())
