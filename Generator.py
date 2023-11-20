class SubGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SubGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.sub_generator1 = SubGenerator(latent_dim, 256)
        self.sub_generator2 = SubGenerator(256, 512)
        self.sub_generator3 = SubGenerator(512, 1024)
        self.sub_generator4 = SubGenerator(1024, 28 * 28)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.sub_generator1(x)
        x = self.sub_generator2(x)
        x = self.sub_generator3(x)
        x = self.sub_generator4(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x
