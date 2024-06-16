# # Import necessary modules
# from argparse import ArgumentParser
# import yaml
# from pytorch_lightning import Trainer
# from gan_module import AgingGAN  # Adjust the import based on your actual module structure
# import torch

# # Create an ArgumentParser object to handle command-line arguments
# parser = ArgumentParser()
# # Add a command-line argument '--config' with a default value and help message
# parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

# # Define the main function
# def main():
#     # Parse the command-line arguments
#     args = parser.parse_args()
    
#     # Open the YAML configuration file specified in the '--config' argument
#     with open(args.config) as file:
#         # Load the YAML content into a dictionary using the PyYAML library
#         config = yaml.load(file, Loader=yaml.FullLoader)
    
#     # Print the loaded configuration for debugging purposes
#     print(config)
    
#     # Create an instance of the AgingGAN model using the loaded configuration
#     model = AgingGAN(config)
    
#     # Create a PyTorch Lightning Trainer instance with specified configurations

#     # trainer = Trainer(max_epochs=config['epochs'], auto_scale_batch_size='binsearch', gpus=1 if torch.cuda.is_available() else 0)
#     trainer = Trainer(max_epochs=config['epochs'], gpus=1 if torch.cuda.is_available() else 0)

    
#     # Train the model using the Trainer instance
#     trainer.fit(model)

# # Entry point of the script
# if __name__ == '__main__':
#     # Call the main function when the script is executed
#     main()

# # import pytorch_lightning as pl
# # print(pl.__version__)
# # import torch
# # print(torch.__version__)
# # print(torch.cuda.is_available())

# import torch
# print(torch.__version__)
# import torch
# print(torch.cuda.is_available())


# Google Colab

# Import necessary modules
# from argparse import ArgumentParser
# import yaml
# from pytorch_lightning import Trainer
# from gan_module import AgingGAN  # Adjust the import based on your actual module structure
# import torch
# # from models import Generator

# # Create an ArgumentParser object to handle command-line arguments
# parser = ArgumentParser()
# # Add a command-line argument '--config' with a default value and help message
# parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

# # Define the main function
# def main():
#     # Parse the command-line arguments
#     args = parser.parse_args()
    
#     # Open the YAML configuration file specified in the '--config' argument
#     with open(args.config) as file:
#         # Load the YAML content into a dictionary using the PyYAML library
#         config = yaml.load(file, Loader=yaml.FullLoader)
    
#     # Print the loaded configuration for debugging purposes
#     print(config)
    
#     # Create an instance of the AgingGAN model using the loaded configuration
#     model = AgingGAN(config)
    
#     # Create a PyTorch Lightning Trainer instance with specified configurations
#     trainer = Trainer(max_epochs=config['epochs'], gpus=1 if torch.cuda.is_available() else 0)
    
#     # Train the model using the Trainer instance
#     trainer.fit(model)

# # Entry point of the script
# if __name__ == '__main__':
#     # Call the main function when the script is executed
#     main()



################################################################################


# # Import necessary modules
# from argparse import ArgumentParser
# import yaml
# from pytorch_lightning import Trainer
# from gan_module import AgingGAN  # Adjust the import based on your actual module structure
# import torch

# # Create an ArgumentParser object to handle command-line arguments
# parser = ArgumentParser()
# # Add a command-line argument '--config' with a default value and help message
# parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')

# # Define the main function
# def main():
#     # Parse the command-line arguments
#     args = parser.parse_args()
    
#     # Open the YAML configuration file specified in the '--config' argument
#     with open(args.config) as file:
#         # Load the YAML content into a dictionary using the PyYAML library
#         config = yaml.load(file, Loader=yaml.FullLoader)
    
#     # Print the loaded configuration for debugging purposes
#     print(config)
    
#     # Create an instance of the AgingGAN model using the loaded configuration
#     model = AgingGAN(config)
    
#     # Create a PyTorch Lightning Trainer instance with specified configurations
#     trainer = Trainer(max_epochs=config['epochs'], gpus=1 if torch.cuda.is_available() else 0)
    
#     # Train the model using the Trainer instance
#     trainer.fit(model)
    
#     # Save model weights
#     model.save_weights('saved_weights')

# # Entry point of the script
# if __name__ == '__main__':
#     # Call the main function when the script is executed
#     main()


#########################################################################


# # Import necessary modules
# from argparse import ArgumentParser
# import yaml
# from pytorch_lightning import Trainer
# from gan_module import AgingGAN  # Adjust the import based on your actual module structure
# from torchsummary import summary
# from torchviz import make_dot
# import torch

# # Create an ArgumentParser object to handle command-line arguments
# parser = ArgumentParser()
# # Add a command-line argument '--config' with a default value and help message
# parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
# parser.add_argument('--weights_dir', default='saved_weights', help='Directory to save model weights')

# # Define the main function
# def main():
#     # Parse the command-line arguments
#     args = parser.parse_args()
    
#     # Open the YAML configuration file specified in the '--config' argument
#     with open(args.config) as file:
#         # Load the YAML content into a dictionary using the PyYAML library
#         config = yaml.load(file, Loader=yaml.FullLoader)
    
#     # Print the loaded configuration for debugging purposes
#     print(config)
    
#     # Create an instance of the AgingGAN model using the loaded configuration
#     model = AgingGAN(config)
    
#     # Print model architecture
#     print("Generator A2B Architecture:")
#     summary(model.genA2B, input_size=(3, 256, 256))
#     print("Generator B2A Architecture:")
#     summary(model.genB2A, input_size=(3, 256, 256))
#     print("Discriminator GA Architecture:")
#     summary(model.disGA, input_size=(3, 256, 256))
#     print("Discriminator GB Architecture:")
#     summary(model.disGB, input_size=(3, 256, 256))
    
#     # Plot model architectures
#     make_dot(model.genA2B(torch.randn(1, 3, 256, 256)), params=dict(model.genA2B.named_parameters())).render("genA2B", format="png")
#     make_dot(model.genB2A(torch.randn(1, 3, 256, 256)), params=dict(model.genB2A.named_parameters())).render("genB2A", format="png")
#     make_dot(model.disGA(torch.randn(1, 3, 256, 256)), params=dict(model.disGA.named_parameters())).render("disGA", format="png")
#     make_dot(model.disGB(torch.randn(1, 3, 256, 256)), params=dict(model.disGB.named_parameters())).render("disGB", format="png")
    
#     # Create a PyTorch Lightning Trainer instance with specified configurations
#     trainer = Trainer(max_epochs=config['epochs'], gpus=1 if torch.cuda.is_available() else 0)
    
#     # Train the model using the Trainer instance
#     trainer.fit(model)
    
#     # Print model weights
#     print("Generator A2B Weights:")
#     print(model.genA2B.state_dict())
#     print("Generator B2A Weights:")
#     print(model.genB2A.state_dict())
#     print("Discriminator GA Weights:")
#     print(model.disGA.state_dict())
#     print("Discriminator GB Weights:")
#     print(model.disGB.state_dict())
    
#     # Save model weights
#     model.save_weights(args.weights_dir)

# # Entry point of the script
# if __name__ == '__main__':
#     # Call the main function when the script is executed
#     main()                             



# Import necessary modules
from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer
from gan_module import AgingGAN  # Adjust the import based on your actual module structure
from torchsummary import summary
from torchviz import make_dot
import torch

# Create an ArgumentParser object to handle command-line arguments
parser = ArgumentParser()
# Add a command-line argument '--config' with a default value and help message
parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
parser.add_argument('--weights_dir', default='saved_weights', help='Directory to save model weights')

# Define the main function
def main():
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Open the YAML configuration file specified in the '--config' argument
    with open(args.config) as file:
        # Load the YAML content into a dictionary using the PyYAML library
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Print the loaded configuration for debugging purposes
    print(config)
    
    # Create an instance of the AgingGAN model using the loaded configuration
    model = AgingGAN(config)
    
    # Print model architecture
    print("Generator A2B Architecture:")
    summary(model.genA2B, input_size=(3, 256, 256))
    print("Generator B2A Architecture:")
    summary(model.genB2A, input_size=(3, 256, 256))
    print("Discriminator GA Architecture:")
    summary(model.disGA, input_size=(3, 256, 256))
    print("Discriminator GB Architecture:")
    summary(model.disGB, input_size=(3, 256, 256))
    
    # Plot model architectures
    make_dot(model.genA2B(torch.randn(1, 3, 256, 256)), params=dict(model.genA2B.named_parameters())).render("genA2B", format="png")
    make_dot(model.genB2A(torch.randn(1, 3, 256, 256)), params=dict(model.genB2A.named_parameters())).render("genB2A", format="png")
    make_dot(model.disGA(torch.randn(1, 3, 256, 256)), params=dict(model.disGA.named_parameters())).render("disGA", format="png")
    make_dot(model.disGB(torch.randn(1, 3, 256, 256)), params=dict(model.disGB.named_parameters())).render("disGB", format="png")
    
    # Create a PyTorch Lightning Trainer instance with specified configurations
    trainer = Trainer(max_epochs=config['epochs'], gpus=1 if torch.cuda.is_available() else 0, logger=False)  # Disable TensorBoard logging
    
    # Train the model using the Trainer instance
    trainer.fit(model)
    
    # Print model weights
    print("Generator A2B Weights:")
    print(model.genA2B.state_dict())
    print("Generator B2A Weights:")
    print(model.genB2A.state_dict())
    print("Discriminator GA Weights:")
    print(model.disGA.state_dict())
    print("Discriminator GB Weights:")
    print(model.disGB.state_dict())
    
    # Save model weights
    model.save_weights(args.weights_dir)

# Entry point of the script
if __name__ == '__main__':
    # Call the main function when the script is executed
    main()