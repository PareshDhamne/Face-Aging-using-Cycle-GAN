# # Import necessary libraries and modules
# import os
# import random
# from argparse import ArgumentParser

# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
# from torchvision import transforms

# # Import the custom Generator class from gan_module
# from gan_module import Generator

# # Create an ArgumentParser to handle command line arguments
# parser = ArgumentParser()
# parser.add_argument(
#     '--image_dir', default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/image', help='The image directory')

# # Decorator to specify that the following function does not need gradient computation
# @torch.no_grad()
# def main():
#     # Parse command line arguments
#     args = parser.parse_args()

#     # Create a list of image paths by joining the directory and file names
#     image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if x.endswith('.png') or x.endswith('.jpg')]
    
#     # Print the number of images found and the list of image paths
#     print("Number of images found:", len(image_paths))
#     print("Image paths:", image_paths)

#     # Create an instance of the Generator class with specified parameters
#     model = Generator(ngf=32, n_residual_blocks=9)

#     # Load pre-trained model weights
#     ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
#     model.load_state_dict(ckpt)

#     # Set the model to evaluation mode
#     model.eval()

#     # Define a series of image transformations using torchvision.transforms
#     trans = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])

#     # Determine the number of images to display (minimum of 6 or the total number of images)
#     nr_images = len(image_paths) if len(image_paths) >= 6 else 6

#     # Create subplots for displaying original and aged faces
#     fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
#     # Shuffle the list of image paths randomly
#     random.shuffle(image_paths)

#     # Loop through the selected number of images
#     for i in range(nr_images):
#         # Open and preprocess the image
#         img = Image.open(image_paths[i]).convert('RGB')
#         img = trans(img).unsqueeze(0)

#         # Generate an aged face using the model
#         aged_face = model(img)

#         # Convert the aged face to numpy array for visualization
#         aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

#         # Display the original and aged faces in the subplots
#         ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
#         ax[1, i].imshow(aged_face)

#     # Save the plot as an image file
#     plt.savefig("mygraph.png")

# # Entry point of the script
# if __name__ == '__main__':
#     main()


#####################################################################


# # Import necessary libraries and modules
# import os
# import random
# from argparse import ArgumentParser

# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
# from torchvision import transforms

# # Import the custom Generator class from gan_module
# from gan_module import Generator

# # Create an ArgumentParser to handle command line arguments
# parser = ArgumentParser()
# parser.add_argument(
#     '--image_dir', default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/image', help='The image directory')
# parser.add_argument(
#     '--weights_dir', default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/saved_weights', help='Directory containing saved model weights')

# # Decorator to specify that the following function does not need gradient computation
# @torch.no_grad()
# def main():
#     # Parse command line arguments
#     args = parser.parse_args()

#     # Create a list of image paths by joining the directory and file names
#     image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if x.endswith('.png') or x.endswith('.jpg')]

#     # Print the number of images found and the list of image paths
#     print("Number of images found:", len(image_paths))
#     print("Image paths:", image_paths)

#     # Create an instance of the Generator class
#     model = Generator(ngf=32, n_residual_blocks=9)

#     # Load the trained model weights
#     model_path = os.path.join(args.weights_dir, 'genA2B.pth')  # Adjust this based on your model naming convention
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

#     # Set the model to evaluation mode
#     model.eval()

#     # Define a series of image transformations
#     trans = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])

#     # Determine the number of images to display
#     nr_images = len(image_paths) if len(image_paths) >= 6 else 6

#     # Create subplots for displaying original and aged faces
#     fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
#     # Shuffle the list of image paths randomly
#     random.shuffle(image_paths)

#     # Loop through the selected number of images
#     for i in range(nr_images):
#         # Open and preprocess the image
#         img = Image.open(image_paths[i]).convert('RGB')
#         img = trans(img).unsqueeze(0)

#         # Generate an aged face using the model
#         aged_face = model(img)

#         # Convert the aged face to numpy array for visualization
#         aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

#         # Display the original and aged faces in the subplots
#         ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
#         ax[1, i].imshow(aged_face)

#     # Save the plot as an image file
#     plt.savefig("mygraph.png")

# # Entry point of the script
# if __name__ == '__main__':
#     main()



# Import necessary libraries and modules
import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

# Import the custom Generator class from gan_module
from gan_module import Generator

# Create an ArgumentParser to handle command line arguments
parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/image', help='The image directory')
parser.add_argument(
    '--weights_dir', default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/saved_weights', help='Directory containing saved model weights')

# Decorator to specify that the following function does not need gradient computation
@torch.no_grad()
def main():
    # Parse command line arguments
    args = parser.parse_args()

    # Create a list of image paths by joining the directory and file names
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if x.endswith('.png') or x.endswith('.jpg')]

    # Print the number of images found and the list of image paths
    print("Number of images found:", len(image_paths))
    print("Image paths:", image_paths)

    # Create an instance of the Generator class
    config = {
        'ngf': 32,
        'n_residual_blocks': 9
    }
    model = Generator(ngf=config['ngf'], n_residual_blocks=config['n_residual_blocks'])

    # Load the trained model weights
    model_path = os.path.join(args.weights_dir, 'genA2B.pth')  # Adjust this based on your model naming convention
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Define a series of image transformations
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Determine the number of images to display
    nr_images = len(image_paths) if len(image_paths) >= 6 else 6

    # Create subplots for displaying original and aged faces
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    # Shuffle the list of image paths randomly
    random.shuffle(image_paths)

    # Loop through the selected number of images
    for i in range(nr_images):
        # Open and preprocess the image
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)

        # Generate an aged face using the model
        aged_face = model(img)

        # Convert the aged face to numpy array for visualization
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

        # Display the original and aged faces in the subplots
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)

    # Save the plot as an image file
    plt.savefig("mygraph.png")

# Entry point of the script
if __name__ == '__main__':
    main()
