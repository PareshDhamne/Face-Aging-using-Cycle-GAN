# # Import necessary libraries
# import itertools
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from pytorch_lightning import LightningModule

# # Import custom modules
# from dataset import ImagetoImageDataset
# from models import Generator, Discriminator

# # Define a LightningModule class named AgingGAN
# class AgingGAN(LightningModule):

#     def __init__(self, hparams):
#         super().__init__()
#         # Save hyperparameters for easy access
#         self._hparams = hparams

#         # Initialize generator and discriminator models
#         self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.disGA = Discriminator(hparams['ndf'])
#         self.disGB = Discriminator(hparams['ndf'])

#         # Cache for generated images
#         self.generated_A = None
#         self.generated_B = None
#         self.real_A = None
#         self.real_B = None

#         # Initialize lists to store losses
#         self.generator_losses = []
#         self.discriminator_losses = []

#     def forward(self, x):
#         return self.genA2B(x)

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # Extract real images from the batch
#         real_A, real_B = batch

#         if optimizer_idx == 0:  # Generator training
#             # Identity loss
#             same_B = self.genA2B(real_B)
#             loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

#             same_A = self.genB2A(real_A)
#             loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

#             # GAN loss
#             fake_B = self.genA2B(real_A)
#             pred_fake = self.disGB(fake_B)
#             loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             fake_A = self.genB2A(real_B)
#             pred_fake = self.disGA(fake_A)
#             loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             # Cycle loss
#             recovered_A = self.genB2A(fake_B)
#             loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

#             recovered_B = self.genA2B(fake_A)
#             loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

#             # Total loss
#             g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

#             # Output dictionary for logging
#             output = {
#                 'loss': g_loss,
#                 'log': {'Loss/Generator': g_loss}
#             }

#             # Cache generated and real images for logging
#             self.generated_B = fake_B
#             self.generated_A = fake_A
#             self.real_B = real_B
#             self.real_A = real_A

#             # Log images and losses to file every 500 batches
#             if batch_idx % 500 == 0:
#                 self.plot_images(batch_idx)

#             return output

#         if optimizer_idx == 1:  # Discriminator training
#             # Real loss for discriminator GA
#             pred_real = self.disGA(real_A)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GA
#             fake_A = self.generated_A
#             pred_fake = self.disGA(fake_A.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GA
#             loss_D_A = (loss_D_real + loss_D_fake) * 0.5

#             # Real loss for discriminator GB
#             pred_real = self.disGB(real_B)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GB
#             fake_B = self.generated_B
#             pred_fake = self.disGB(fake_B.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GB
#             loss_D_B = (loss_D_real + loss_D_fake) * 0.5

#             # Total discriminator loss
#             d_loss = loss_D_A + loss_D_B
#             output = {
#                 'loss': d_loss,
#                 'log': {'Loss/Discriminator': d_loss}
#             }

#             # Append discriminator loss to the list
#             self.discriminator_losses.append(d_loss.item())

#             return output

#     def configure_optimizers(self):
#         # Define optimizers for generator and discriminator
#         g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         return [g_optim, d_optim], []

#     def train_dataloader(self):
#         # Define image transformations for training data
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
#             transforms.RandomCrop(self.hparams['img_size']),
#             transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])

#         # Create dataset using the defined transformations
#         dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)

#         # Create DataLoader with specified batch size and number of workers
#         return DataLoader(dataset,
#                           batch_size=self.hparams['batch_size'],
#                           num_workers=self.hparams['num_workers'],
#                           shuffle=True)

#     def plot_images(self, batch_idx):
#         # Plot and save images
#         plt.figure(figsize=(10, 10))

#         # Plot real images
#         plt.subplot(2, 2, 1)
#         plt.imshow(make_grid(self.real_A, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 2)
#         plt.imshow(make_grid(self.real_B, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/B')
#         plt.axis('off')

#         # Plot generated images
#         plt.subplot(2, 2, 3)
#         plt.imshow(make_grid(self.generated_A, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 4)
#         plt.imshow(make_grid(self.generated_B, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/B')
#         plt.axis('off')

#         plt.savefig(f'images_epoch_{self.current_epoch}_batch_{batch_idx}.png')
#         plt.close()

#         # Plot losses
#         self.plot_losses()

#     def plot_losses(self):
#         plt.figure(figsize=(10, 5))

#         # Plot generator loss
#         plt.subplot(1, 2, 1)
#         plt.plot(self.generator_losses, label='Generator Loss')
#         plt.title('Generator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         # Plot discriminator loss
#         plt.subplot(1, 2, 2)
#         plt.plot(self.discriminator_losses, label='Discriminator Loss')
#         plt.title('Discriminator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(f'losses_plot_epoch_{self.current_epoch}.png')
#         plt.close()

# Google Colab

# import itertools
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from pytorch_lightning import LightningModule
# from dataset import ImagetoImageDataset
# from models import Generator, Discriminator

# class AgingGAN(LightningModule):

#     def __init__(self, hparams):
#         super().__init__()
#         # Save hyperparameters for easy access
#         self._hparams = hparams

#         # Initialize generator and discriminator models
#         self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.disGA = Discriminator(hparams['ndf'])
#         self.disGB = Discriminator(hparams['ndf'])

#         # Cache for generated images
#         self.generated_A = None
#         self.generated_B = None
#         self.real_A = None
#         self.real_B = None

#         # Initialize lists to store losses
#         self.generator_losses = []
#         self.discriminator_losses = []

#     def forward(self, x):
#         return self.genA2B(x)

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # Extract real images from the batch
#         real_A, real_B = batch

#         if optimizer_idx == 0:  # Generator training
#             # Identity loss
#             same_B = self.genA2B(real_B)
#             loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

#             same_A = self.genB2A(real_A)
#             loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

#             # GAN loss
#             fake_B = self.genA2B(real_A)
#             pred_fake = self.disGB(fake_B)
#             loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             fake_A = self.genB2A(real_B)
#             pred_fake = self.disGA(fake_A)
#             loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             # Cycle loss
#             recovered_A = self.genB2A(fake_B)
#             loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

#             recovered_B = self.genA2B(fake_A)
#             loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

#             # Total loss
#             g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

#             # Output dictionary for logging
#             output = {
#                 'loss': g_loss,
#                 'log': {'Loss/Generator': g_loss}
#             }

#             # Cache generated and real images for logging
#             self.generated_B = fake_B
#             self.generated_A = fake_A
#             self.real_B = real_B
#             self.real_A = real_A

#             # Log images and losses to file every 500 batches
#             if batch_idx % 500 == 0:
#                 self.plot_images(batch_idx)

#             return output

#         if optimizer_idx == 1:  # Discriminator training
#             # Real loss for discriminator GA
#             pred_real = self.disGA(real_A)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GA
#             fake_A = self.generated_A
#             pred_fake = self.disGA(fake_A.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GA
#             loss_D_A = (loss_D_real + loss_D_fake) * 0.5

#             # Real loss for discriminator GB
#             pred_real = self.disGB(real_B)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GB
#             fake_B = self.generated_B
#             pred_fake = self.disGB(fake_B.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GB
#             loss_D_B = (loss_D_real + loss_D_fake) * 0.5

#             # Total discriminator loss
#             d_loss = loss_D_A + loss_D_B
#             output = {
#                 'loss': d_loss,
#                 'log': {'Loss/Discriminator': d_loss}
#             }

#             # Append discriminator loss to the list
#             self.discriminator_losses.append(d_loss.item())

#             return output

#     def configure_optimizers(self):
#         # Define optimizers for generator and discriminator
#         g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         return [g_optim, d_optim], []

#     def train_dataloader(self):
#         # Define image transformations for training data
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
#             transforms.RandomCrop(self.hparams['img_size']),
#             transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])

#         # Create dataset using the defined transformations
#         dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)

#         # Create DataLoader with specified batch size and number of workers
#         return DataLoader(dataset,
#                           batch_size=self.hparams['batch_size'],
#                           num_workers=self.hparams['num_workers'],
#                           shuffle=True)

#     def plot_images(self, batch_idx):
#         # Move tensors to CPU before plotting
#         real_A_cpu = self.real_A.cpu()
#         real_B_cpu = self.real_B.cpu()
#         generated_A_cpu = self.generated_A.cpu()
#         generated_B_cpu = self.generated_B.cpu()

#         # Plot and save images
#         plt.figure(figsize=(10, 10))

#         # Plot real images
#         plt.subplot(2, 2, 1)
#         plt.imshow(make_grid(real_A_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 2)
#         plt.imshow(make_grid(real_B_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/B')
#         plt.axis('off')

#         # Plot generated images
#         plt.subplot(2, 2, 3)
#         plt.imshow(make_grid(generated_A_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 4)
#         plt.imshow(make_grid(generated_B_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/B')
#         plt.axis('off')

#         plt.savefig(f'images_epoch_{self.current_epoch}_batch_{batch_idx}.png')
#         plt.close()

#         # Plot losses
#         self.plot_losses()

#     def plot_losses(self):
#         plt.figure(figsize=(10, 5))

#         # Plot generator loss
#         plt.subplot(1, 2, 1)
#         plt.plot(self.generator_losses, label='Generator Loss')
#         plt.title('Generator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         # Plot discriminator loss
#         plt.subplot(1, 2, 2)
#         plt.plot(self.discriminator_losses, label='Discriminator Loss')
#         plt.title('Discriminator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(f'losses_plot_epoch_{self.current_epoch}.png')
#         plt.close()


# Plotting...


# # Import necessary libraries
# import itertools
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from pytorch_lightning import LightningModule

# # Import custom modules
# from dataset import ImagetoImageDataset
# from models import Generator, Discriminator

# # Define a LightningModule class named AgingGAN
# class AgingGAN(LightningModule):

#     def __init__(self, hparams):
#         super().__init__()
#         # Save hyperparameters for easy access
#         self._hparams = hparams

#         # Initialize generator and discriminator models
#         self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.disGA = Discriminator(hparams['ndf'])
#         self.disGB = Discriminator(hparams['ndf'])

#         # Cache for generated images
#         self.generated_A = None
#         self.generated_B = None
#         self.real_A = None
#         self.real_B = None

#         # Initialize lists to store losses
#         self.generator_losses = []
#         self.discriminator_losses = []

#     def forward(self, x):
#         return self.genA2B(x)

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # Extract real images from the batch
#         real_A, real_B = batch

#         if optimizer_idx == 0:  # Generator training
#             # Identity loss
#             same_B = self.genA2B(real_B)
#             loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

#             same_A = self.genB2A(real_A)
#             loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

#             # GAN loss
#             fake_B = self.genA2B(real_A)
#             pred_fake = self.disGB(fake_B)
#             loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             fake_A = self.genB2A(real_B)
#             pred_fake = self.disGA(fake_A)
#             loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             # Cycle loss
#             recovered_A = self.genB2A(fake_B)
#             loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

#             recovered_B = self.genA2B(fake_A)
#             loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

#             # Total loss
#             g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

#             # Append generator loss to the list
#             self.generator_losses.append(g_loss.item())

#             # Output dictionary for logging
#             output = {
#                 'loss': g_loss,
#                 'log': {'Loss/Generator': g_loss}
#             }

#             # Cache generated and real images for logging
#             self.generated_B = fake_B
#             self.generated_A = fake_A
#             self.real_B = real_B
#             self.real_A = real_A

#             # Log images and losses to file every 500 batches
#             if batch_idx % 500 == 0:
#                 self.plot_images(batch_idx)

#             return output

#         if optimizer_idx == 1:  # Discriminator training
#             # Real loss for discriminator GA
#             pred_real = self.disGA(real_A)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GA
#             fake_A = self.generated_A
#             pred_fake = self.disGA(fake_A.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GA
#             loss_D_A = (loss_D_real + loss_D_fake) * 0.5

#             # Real loss for discriminator GB
#             pred_real = self.disGB(real_B)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GB
#             fake_B = self.generated_B
#             pred_fake = self.disGB(fake_B.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GB
#             loss_D_B = (loss_D_real + loss_D_fake) * 0.5

#             # Total discriminator loss
#             d_loss = loss_D_A + loss_D_B
#             output = {
#                 'loss': d_loss,
#                 'log': {'Loss/Discriminator': d_loss}
#             }

#             # Append discriminator loss to the list
#             self.discriminator_losses.append(d_loss.item())

#             return output

#     def configure_optimizers(self):
#         # Define optimizers for generator and discriminator
#         g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         return [g_optim, d_optim], []

#     def train_dataloader(self):
#         # Define image transformations for training data
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
#             transforms.RandomCrop(self.hparams['img_size']),
#             transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])

#         # Create dataset using the defined transformations
#         dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)

#         # Create DataLoader with specified batch size and number of workers
#         return DataLoader(dataset,
#                           batch_size=self.hparams['batch_size'],
#                           num_workers=self.hparams['num_workers'],
#                           shuffle=True)

#     def plot_images(self, batch_idx):
#         # Plot and save images
#         plt.figure(figsize=(10, 10))

#         # Plot real images
#         plt.subplot(2, 2, 1)
#         plt.imshow(make_grid(self.real_A, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 2)
#         plt.imshow(make_grid(self.real_B, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/B')
#         plt.axis('off')

#         # Plot generated images
#         plt.subplot(2, 2, 3)
#         plt.imshow(make_grid(self.generated_A, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 4)
#         plt.imshow(make_grid(self.generated_B, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/B')
#         plt.axis('off')

#         plt.savefig(f'images_epoch_{self.current_epoch}_batch_{batch_idx}.png')
#         plt.close()

#         # Plot losses
#         self.plot_losses()

#     def plot_losses(self):
#         plt.figure(figsize=(10, 5))

#         # Plot generator loss
#         plt.subplot(1, 2, 1)
#         plt.plot(range(len(self.generator_losses)), self.generator_losses, label='Generator Loss')
#         plt.title('Generator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         # Plot discriminator loss
#         plt.subplot(1, 2, 2)
#         plt.plot(range(len(self.discriminator_losses)), self.discriminator_losses, label='Discriminator Loss')
#         plt.title('Discriminator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(f'losses_plot_epoch_{self.current_epoch}.png')
#         plt.close()

# Google Colab

# import itertools
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# from pytorch_lightning import LightningModule
# from dataset import ImagetoImageDataset
# from models import Generator, Discriminator

# class AgingGAN(LightningModule):

#     def __init__(self, hparams):
#         super().__init__()
#         # Save hyperparameters for easy access
#         self._hparams = hparams

#         # Initialize generator and discriminator models
#         self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
#         self.disGA = Discriminator(hparams['ndf'])
#         self.disGB = Discriminator(hparams['ndf'])

#         # Cache for generated images
#         self.generated_A = None
#         self.generated_B = None
#         self.real_A = None
#         self.real_B = None

#         # Initialize lists to store losses
#         self.generator_losses = []
#         self.discriminator_losses = []

#     def forward(self, x):
#         return self.genA2B(x)

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # Extract real images from the batch
#         real_A, real_B = batch

#         if optimizer_idx == 0:  # Generator training
#             # Identity loss
#             same_B = self.genA2B(real_B)
#             loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

#             same_A = self.genB2A(real_A)
#             loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

#             # GAN loss
#             fake_B = self.genA2B(real_A)
#             pred_fake = self.disGB(fake_B)
#             loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             fake_A = self.genB2A(real_B)
#             pred_fake = self.disGA(fake_A)
#             loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

#             # Cycle loss
#             recovered_A = self.genB2A(fake_B)
#             loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

#             recovered_B = self.genA2B(fake_A)
#             loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

#             # Total loss
#             g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

#             # Append generator loss to the list
#             self.generator_losses.append(g_loss.item())

#             # Output dictionary for logging
#             output = {
#                 'loss': g_loss,
#                 'log': {'Loss/Generator': g_loss}
#             }

#             # Cache generated and real images for logging
#             self.generated_B = fake_B
#             self.generated_A = fake_A
#             self.real_B = real_B
#             self.real_A = real_A

#             # Log images and losses to file every 500 batches
#             if batch_idx % 500 == 0:
#                 self.plot_images(batch_idx)

#             return output

#         if optimizer_idx == 1:  # Discriminator training
#             # Real loss for discriminator GA
#             pred_real = self.disGA(real_A)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GA
#             fake_A = self.generated_A
#             pred_fake = self.disGA(fake_A.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GA
#             loss_D_A = (loss_D_real + loss_D_fake) * 0.5

#             # Real loss for discriminator GB
#             pred_real = self.disGB(real_B)
#             loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

#             # Fake loss for discriminator GB
#             fake_B = self.generated_B
#             pred_fake = self.disGB(fake_B.detach())
#             loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

#             # Total loss for discriminator GB
#             loss_D_B = (loss_D_real + loss_D_fake) * 0.5

#             # Total discriminator loss
#             d_loss = loss_D_A + loss_D_B
#             output = {
#                 'loss': d_loss,
#                 'log': {'Loss/Discriminator': d_loss}
#             }

#             # Append discriminator loss to the list
#             self.discriminator_losses.append(d_loss.item())

#             return output

#     def configure_optimizers(self):
#         # Define optimizers for generator and discriminator
#         g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
#                                    lr=self.hparams['lr'], betas=(0.5, 0.999),
#                                    weight_decay=self.hparams['weight_decay'])
#         return [g_optim, d_optim], []

#     def train_dataloader(self):
#         # Define image transformations for training data
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
#             transforms.RandomCrop(self.hparams['img_size']),
#             transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])

#         # Create dataset using the defined transformations
#         dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)

#         # Create DataLoader with specified batch size and number of workers
#         return DataLoader(dataset,
#                           batch_size=self.hparams['batch_size'],
#                           num_workers=self.hparams['num_workers'],
#                           shuffle=True)

#     def plot_images(self, batch_idx):
#         # Move tensors to CPU before plotting
#         real_A_cpu = self.real_A.cpu()
#         real_B_cpu = self.real_B.cpu()
#         generated_A_cpu = self.generated_A.cpu()
#         generated_B_cpu = self.generated_B.cpu()

#         # Plot and save images
#         plt.figure(figsize=(10, 10))

#         # Plot real images
#         plt.subplot(2, 2, 1)
#         plt.imshow(make_grid(real_A_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 2)
#         plt.imshow(make_grid(real_B_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Real/B')
#         plt.axis('off')

#         # Plot generated images
#         plt.subplot(2, 2, 3)
#         plt.imshow(make_grid(generated_A_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/A')
#         plt.axis('off')

#         plt.subplot(2, 2, 4)
#         plt.imshow(make_grid(generated_B_cpu, normalize=True, scale_each=True).permute(1, 2, 0))
#         plt.title('Generated/B')
#         plt.axis('off')

#         plt.savefig(f'images_epoch_{self.current_epoch}_batch_{batch_idx}.png')
#         plt.close()

#         # Plot losses
#         self.plot_losses()

#     def plot_losses(self):
#         plt.figure(figsize=(10, 5))

#         # Plot generator loss
#         plt.subplot(1, 2, 1)
#         plt.plot(self.generator_losses, label='Generator Loss')
#         plt.title('Generator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         # Plot discriminator loss
#         plt.subplot(1, 2, 2)
#         plt.plot(self.discriminator_losses, label='Discriminator Loss')
#         plt.title('Discriminator Loss')
#         plt.xlabel('Batch Index')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig(f'losses_plot_epoch_{self.current_epoch}.png')
#         plt.close()

# Instantiate your model
# hparams = {'ngf': 64, 'n_blocks': 6, 'ndf': 64, 'identity_weight': 5.0, 'adv_weight': 1.0, 'cycle_weight': 10.0,
#            'lr': 0.0002, 'weight_decay': 0.0, 'img_size': 256, 'augment_rotation': 10, 'batch_size': 8,
#            'num_workers': 4, 'domainA_dir': 'path_to_domainA_images', 'domainB_dir': 'path_to_domainB_images'}
# model = AgingGAN(hparams)

# Train your model
# trainer = Trainer(max_epochs=1000)
# trainer.fit(model)



#############################################################

# Import necessary libraries
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from pytorch_lightning import LightningModule

# Import custom modules
from dataset import ImagetoImageDataset
from models import Generator, Discriminator

# Define a LightningModule class named AgingGAN
class AgingGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # Save hyperparameters for easy access
        self._hparams = hparams

        # Initialize generator and discriminator models
        self.genA2B = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.genB2A = Generator(hparams['ngf'], n_residual_blocks=hparams['n_blocks'])
        self.disGA = Discriminator(hparams['ndf'])
        self.disGB = Discriminator(hparams['ndf'])

        # Cache for generated images
        self.generated_A = None
        self.generated_B = None
        self.real_A = None
        self.real_B = None

        # Initialize lists to store losses
        self.generator_losses = []
        self.discriminator_losses = []

    def forward(self, x):
        return self.genA2B(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Extract real images from the batch
        real_A, real_B = batch

        if optimizer_idx == 0:  # Generator training
            # Identity loss
            same_B = self.genA2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * self.hparams['identity_weight']

            same_A = self.genB2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * self.hparams['identity_weight']

            # GAN loss
            fake_B = self.genA2B(real_A)
            pred_fake = self.disGB(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

            fake_A = self.genB2A(real_B)
            pred_fake = self.disGA(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, torch.ones_like(pred_fake)) * self.hparams['adv_weight']

            # Cycle loss
            recovered_A = self.genB2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * self.hparams['cycle_weight']

            recovered_B = self.genA2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * self.hparams['cycle_weight']

            # Total loss
            g_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Append generator loss to the list
            self.generator_losses.append(g_loss.item())

            # Output dictionary for logging
            output = {
                'loss': g_loss,
                'log': {'Loss/Generator': g_loss}
            }

            # Cache generated and real images for logging
            self.generated_B = fake_B
            self.generated_A = fake_A
            self.real_B = real_B
            self.real_A = real_A

            # Log images and losses to file every 500 batches
            if batch_idx % 500 == 0:
                self.plot_images(batch_idx)

            return output

        if optimizer_idx == 1:  # Discriminator training
            # Real loss for discriminator GA
            pred_real = self.disGA(real_A)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

            # Fake loss for discriminator GA
            fake_A = self.generated_A
            pred_fake = self.disGA(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

            # Total loss for discriminator GA
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # Real loss for discriminator GB
            pred_real = self.disGB(real_B)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))

            # Fake loss for discriminator GB
            fake_B = self.generated_B
            pred_fake = self.disGB(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))

            # Total loss for discriminator GB
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            # Total discriminator loss
            d_loss = loss_D_A + loss_D_B
            output = {
                'loss': d_loss,
                'log': {'Loss/Discriminator': d_loss}
            }

            # Append discriminator loss to the list
            self.discriminator_losses.append(d_loss.item())

            return output

    def configure_optimizers(self):
        # Define optimizers for generator and discriminator
        g_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        d_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
                                   lr=self.hparams['lr'], betas=(0.5, 0.999),
                                   weight_decay=self.hparams['weight_decay'])
        return [g_optim, d_optim], []

    def train_dataloader(self):
        # Define image transformations for training data
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.hparams['img_size'] + 50, self.hparams['img_size'] + 50)),
            transforms.RandomCrop(self.hparams['img_size']),
            transforms.RandomRotation(degrees=(0, int(self.hparams['augment_rotation']))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # Create dataset using the defined transformations
        dataset = ImagetoImageDataset(self.hparams['domainA_dir'], self.hparams['domainB_dir'], train_transform)

        # Create DataLoader with specified batch size and number of workers
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)

    def plot_images(self, batch_idx):
        # Plot and save images
        plt.figure(figsize=(10, 10))

        # Plot real images
        plt.subplot(2, 2, 1)
        plt.imshow(make_grid(self.real_A, normalize=True, scale_each=True).permute(1, 2, 0))
        plt.title('Real/A')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(make_grid(self.real_B, normalize=True, scale_each=True).permute(1, 2, 0))
        plt.title('Real/B')
        plt.axis('off')

        # Plot generated images
        plt.subplot(2, 2, 3)
        plt.imshow(make_grid(self.generated_A, normalize=True, scale_each=True).permute(1, 2, 0))
        plt.title('Generated/A')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(make_grid(self.generated_B, normalize=True, scale_each=True).permute(1, 2, 0))
        plt.title('Generated/B')
        plt.axis('off')

        plt.savefig(f'images_epoch_{self.current_epoch}_batch_{batch_idx}.png')
        plt.close()

        # Plot losses
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))

        # Plot generator loss
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.generator_losses)), self.generator_losses, label='Generator Loss')
        plt.title('Generator Loss')
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.legend()

        # Plot discriminator loss
        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.discriminator_losses)), self.discriminator_losses, label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Batch Index')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'losses_plot_epoch_{self.current_epoch}.png')
        plt.close()

    def save_weights(self, directory):
        """Save the weights of the models to the specified directory."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.genA2B.state_dict(), os.path.join(directory, 'genA2B.pth'))
        torch.save(self.genB2A.state_dict(), os.path.join(directory, 'genB2A.pth'))
        torch.save(self.disGA.state_dict(), os.path.join(directory, 'disGA.pth'))
        torch.save(self.disGB.state_dict(), os.path.join(directory, 'disGB.pth'))