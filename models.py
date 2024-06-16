# # Import necessary modules from PyTorch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define a ResidualBlock class that inherits from nn.Module
# class ResidualBlock(nn.Module):
#     # Constructor method, takes in the number of input features
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()

#         # Define a convolutional block with two 3x3 convolutions and batch normalization
#         conv_block = [nn.ReflectionPad2d(1),
#                       nn.Conv2d(in_features, in_features, 3),
#                       nn.BatchNorm2d(in_features),
#                       nn.ReLU(),
#                       nn.ReflectionPad2d(1),
#                       nn.Conv2d(in_features, in_features, 3),
#                       nn.BatchNorm2d(in_features)]

#         # Create a sequential module with the defined convolutional block
#         self.conv_block = nn.Sequential(*conv_block)

#     # Forward method for the residual block
#     def forward(self, x):
#         # Return the input added to the output of the convolutional block
#         return x + self.conv_block(x)

# # Define a Generator class that inherits from nn.Module
# class Generator(nn.Module):
#     # Constructor method, takes in the number of generator features (ngf) and the number of residual blocks
#     def __init__(self, ngf, n_residual_blocks=9):
#         super(Generator, self).__init__()

#         # Initial convolution block with reflection padding, convolution, batch normalization, and ReLU activation
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(3, ngf, 7),
#                  nn.BatchNorm2d(ngf),
#                  nn.ReLU()]

#         # Downsampling: Convolution, batch normalization, and ReLU activation with a stride of 2
#         in_features = ngf
#         out_features = in_features * 2
#         for _ in range(2):
#             model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                       nn.BatchNorm2d(out_features),
#                       nn.ReLU()]
#             in_features = out_features
#             out_features = in_features * 2

#         # Add residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]

#         # Upsampling: Transposed convolution, batch normalization, and ReLU activation with a stride of 2
#         out_features = in_features // 2
#         for _ in range(2):
#             model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                       nn.BatchNorm2d(out_features),
#                       nn.ReLU()]
#             in_features = out_features
#             out_features = in_features // 2

#         # Output layer with reflection padding, convolution, and Tanh activation
#         model += [nn.ReflectionPad2d(3),
#                   nn.Conv2d(ngf, 3, 7),
#                   nn.Tanh()]

#         # Create a sequential module with the defined generator model
#         self.model = nn.Sequential(*model)

#     # Forward method for the generator
#     def forward(self, x):
#         # Return the output of the generator model
#         return self.model(x)

# # Define a Discriminator class that inherits from nn.Module
# class Discriminator(nn.Module):
#     # Constructor method, takes in the number of discriminator features (ndf)
#     def __init__(self, ndf):
#         super(Discriminator, self).__init__()

#         # Convolutional layers with leaky ReLU activation for discriminator
#         model = [nn.Conv2d(3, ndf, 4, stride=2, padding=1),
#                  nn.LeakyReLU(0.2, inplace=True)]

#         model += [nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
#                   nn.BatchNorm2d(ndf * 2),
#                   nn.LeakyReLU(0.2, inplace=True)]

#         model += [nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
#                   nn.InstanceNorm2d(ndf * 4),
#                   nn.LeakyReLU(0.2, inplace=True)]

#         model += [nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
#                   nn.InstanceNorm2d(ndf * 8),
#                   nn.LeakyReLU(0.2, inplace=True)]

#         # Fully connected-like classification layer
#         model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]

#         # Create a sequential module with the defined discriminator model
#         self.model = nn.Sequential(*model)

#     # Forward method for the discriminator
#     def forward(self, x):
#         # Pass input through the discriminator model
#         x = self.model(x)
#         # Apply average pooling and flatten the output
#         return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



# Import necessary modules from PyTorch
import torch.nn as nn
import torch.nn.functional as F

# Define a ResidualBlock class that inherits from nn.Module
class ResidualBlock(nn.Module):
    # Constructor method, takes in the number of input features
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # Define a convolutional block with two 3x3 convolutions and batch normalization
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.BatchNorm2d(in_features)]

        # Create a sequential module with the defined convolutional block
        self.conv_block = nn.Sequential(*conv_block)

    # Forward method for the residual block
    def forward(self, x):
        # Return the input added to the output of the convolutional block
        return x + self.conv_block(x)

# Define a Generator class that inherits from nn.Module
class Generator(nn.Module):
    # Constructor method, takes in the number of generator features (ngf) and the number of residual blocks
    def __init__(self, ngf, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block with reflection padding, convolution, batch normalization, and ReLU activation
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, 7),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU()]

        # Downsampling: Convolution, batch normalization, and ReLU activation with a stride of 2
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features * 2

        # Add residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling: Transposed convolution, batch normalization, and ReLU activation with a stride of 2
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(out_features),
                      nn.ReLU()]
            in_features = out_features
            out_features = in_features // 2

        # Output layer with reflection padding, convolution, and Tanh activation
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, 3, 7),
                  nn.Tanh()]

        # Create a sequential module with the defined generator model
        self.model = nn.Sequential(*model)

    # Forward method for the generator
    def forward(self, x):
        # Return the output of the generator model
        return self.model(x)

# Define a Discriminator class that inherits from nn.Module
class Discriminator(nn.Module):
    # Constructor method, takes in the number of discriminator features (ndf)
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        # Convolutional layers with leaky ReLU activation for discriminator
        model = [nn.Conv2d(3, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
                  nn.BatchNorm2d(ndf * 2),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(ndf * 4),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf * 4, ndf * 8, 4, padding=1),
                  nn.InstanceNorm2d(ndf * 8),
                  nn.LeakyReLU(0.2, inplace=True)]

        # Fully connected-like classification layer
        model += [nn.Conv2d(ndf * 8, 1, 4, padding=1)]

        # Create a sequential module with the defined discriminator model
        self.model = nn.Sequential(*model)

    # Forward method for the discriminator
    def forward(self, x):
        # Pass input through the discriminator model
        x = self.model(x)
        # Apply average pooling and flatten the output
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# wrinkles



