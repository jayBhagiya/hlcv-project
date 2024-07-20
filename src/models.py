import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, img_shape):
        super(Generator, self).__init__()

        self.C, self.H, self.W = img_shape
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: [3, 368, 544]
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # output: [64, 368, 544]
        # self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: [64, 368, 544]
        self.b1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [64, 184, 272]

        # input: [64, 187, 275]
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: [128, 184, 272]
        # self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: [128, 184, 272]
        self.b2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [128, 92, 136]

        # input: [128, 93, 137]
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: [256, 92, 136]
        # self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: [256, 92, 136]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [256, 46, 68]

        # input: [256, 46, 68]
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: [512, 46, 68]
        # self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: [512, 46, 68]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [512, 23, 34]

        # input: [512, 23, 34]
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: [1024, 23, 34]
        # self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: [1024, 23, 34]


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, self.C, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        # xe12 = F.relu(self.e12(xe11))
        # xp1 = self.pool1(xe12)
        xp1 = self.pool1(xe11)

        xe21 = F.relu(self.e21(xp1))
        # xe22 = F.relu(self.e22(xe21))
        # xp2 = self.pool2(xe22)
        xp2 = self.pool2(xe21)

        xe31 = F.relu(self.e31(xp2))
        # xe32 = F.relu(self.e32(xe31))
        # xp3 = self.pool3(xe32)
        xp3 = self.pool3(xe31)

        xe41 = F.relu(self.e41(xp3))
        # xe42 = F.relu(self.e42(xe41))
        # xp4 = self.pool4(xe42)
        xp4 = self.pool4(xe41)

        xe51 = F.relu(self.e51(xp4))
        # xe52 = F.relu(self.e52(xe51))

        # # Decoder
        xu1 = self.upconv1(xe51)
        xu11 = torch.cat([xu1, xe41], dim=1)
        xd11 = F.relu(self.d11(xu11))
        # xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd11)
        xu22 = torch.cat([xu2, xe31], dim=1)
        xd21 = F.relu(self.d21(xu22))
        # xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd21)
        xu33 = torch.cat([xu3, xe21], dim=1)
        xd31 = F.relu(self.d31(xu33))
        # xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd31)
        xu44 = torch.cat([xu4, xe11], dim=1)
        xd41 = F.relu(self.d41(xu44))
        # xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd41)

        return out

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.C, self.H, self.W = img_shape
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: [3, 368, 544]
        self.conv1 = nn.Conv2d(3*2, 32, kernel_size=3, stride = 2, padding=1, padding_mode="reflect") # output: [32, 184, 272]

        # input: [32, 184, 272]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1, padding_mode="reflect") # output: [64, 92, 136]
        self.b2 = nn.BatchNorm2d(64)

        # input: [64, 92, 136]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride = 2,padding=1, padding_mode="reflect") # output: [128, 46, 68]
        self.b3 = nn.BatchNorm2d(128)

        # input: [128, 46, 68]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding=1,padding_mode="reflect") # output: [256, 23, 34]
        self.b4 = nn.BatchNorm2d(256)

        # input: [256, 23, 34]
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride = 1,padding=1, padding_mode="reflect") # output: [512, 23, 34]
        self.b5 = nn.BatchNorm2d(512)

        # input: [512, 23, 34]
        self.conv6 = nn.Conv2d(512, 1, kernel_size=3, stride =1, padding = 1, padding_mode="reflect") # output: [1, 23, 34]

    def forward(self, x, y):
        cat = torch.cat([x,y], dim=1)
        x = F.leaky_relu(self.conv1(cat), 0.2)
        x = F.leaky_relu(self.b2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.b3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.b4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.b5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        return x
