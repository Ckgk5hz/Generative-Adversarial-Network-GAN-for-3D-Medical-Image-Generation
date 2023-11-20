class Discriminator(nn.Module):
    def __init__(self, channel=32):
        super(Discriminator, self).__init__()
        _c = channel

        self.conv1 = nn.Conv3d(1, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c)

        self.conv2 = nn.Conv3d(_c, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*2)

        self.conv3 = nn.Conv3d(_c*2, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*4)

        self.conv4 = nn.Conv3d(_c*4, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*8)

        self.conv5 = nn.Conv3d(_c*8, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*16)

        self.conv6 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c*16)

        self.conv7 = nn.Conv3d(_c*16, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(self.bn1(x))

        x = self.conv2(x)
        x = self.leaky_relu(self.bn2(x))

        x = self.conv3(x)
        x = self.leaky_relu(self.bn3(x))

        x = self.conv4(x)
        x = self.leaky_relu(self.bn4(x))

        x = self.conv5(x)
        x = self.leaky_relu(self.bn5(x))

        x = self.conv6(x)
        x = self.leaky_relu(self.bn6(x))

        x = self.conv7(x)
        x.sigmoid_()

        return x
