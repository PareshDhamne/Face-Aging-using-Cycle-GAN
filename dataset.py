# # Import necessary libraries
# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import numpy as np

# # Define a list of image file extensions that the dataset should consider
# IMG_EXTENSIONS = ["png", "jpg"]

# # Define a custom dataset class named ImagetoImageDataset that inherits from torch.utils.data.Dataset
# class ImagetoImageDataset(Dataset):
#     # Constructor method to initialize the dataset with the paths to images in two domains (domainA and domainB)
#     def __init__(self, domainA_dir, domainB_dir, transforms=None):
#         # Create a list of file paths for images in domainA with specified extensions
#         self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if
#                         x.lower().endswith(tuple(IMG_EXTENSIONS))]
#         # Create a list of file paths for images in domainB with specified extensions
#         self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
#                         x.lower().endswith(tuple(IMG_EXTENSIONS))]

#         # Store the transformation function to be applied on images if provided
#         self.transforms = transforms

#         # Store the lengths of the image lists for domains A and B
#         self.lenA = len(self.imagesA)
#         self.lenB = len(self.imagesB)

#     # Method to get the length of the dataset
#     def __len__(self):
#         # Return the maximum length of the two domains
#         return max(self.lenA, self.lenB)

#     # Method to get an item from the dataset given an index (idx)
#     def __getitem__(self, idx):
#         # Initialize indices for domainA (idx_a) and domainB (idx_b) to the given index
#         idx_a = idx_b = idx
#         # If the index exceeds the length of domainA, randomly select an index within the range
#         if idx_a >= self.lenA:
#             idx_a = np.random.randint(self.lenA)
#         # If the index exceeds the length of domainB, randomly select an index within the range
#         if idx_b >= self.lenB:
#             idx_b = np.random.randint(self.lenB)

#         # Read and convert the images from domainA and domainB to RGB arrays
#         imageA = np.array(Image.open(self.imagesA[idx_a]).convert("RGB"))
#         imageB = np.array(Image.open(self.imagesB[idx_b]).convert("RGB"))

#         # Apply transformations to the images if a transformation function is provided
#         if self.transforms is not None:
#             imageA = self.transforms(imageA)
#             imageB = self.transforms(imageB)

#         # Return the transformed or original images from domainA and domainB
#         return imageA, imageB




# Import necessary libraries
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Define a list of image file extensions that the dataset should consider
IMG_EXTENSIONS = ["png", "jpg"]

# Define a custom dataset class named ImagetoImageDataset that inherits from torch.utils.data.Dataset
class ImagetoImageDataset(Dataset):
    # Constructor method to initialize the dataset with the paths to images in two domains (domainA and domainB)
    def __init__(self, domainA_dir, domainB_dir, transforms=None):
        # Create a list of file paths for images in domainA with specified extensions
        self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if
                        x.lower().endswith(tuple(IMG_EXTENSIONS))]
        # Create a list of file paths for images in domainB with specified extensions
        self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
                        x.lower().endswith(tuple(IMG_EXTENSIONS))]

        # Store the transformation function to be applied on images if provided
        self.transforms = transforms

        # Store the lengths of the image lists for domains A and B
        self.lenA = len(self.imagesA)
        self.lenB = len(self.imagesB)

    # Method to get the length of the dataset
    def __len__(self):
        # Return the maximum length of the two domains
        return max(self.lenA, self.lenB)

    # Method to get an item from the dataset given an index (idx)
    def __getitem__(self, idx):
        # Initialize indices for domainA (idx_a) and domainB (idx_b) to the given index
        idx_a = idx_b = idx
        # If the index exceeds the length of domainA, randomly select an index within the range
        if idx_a >= self.lenA:
            idx_a = np.random.randint(self.lenA)
        # If the index exceeds the length of domainB, randomly select an index within the range
        if idx_b >= self.lenB:
            idx_b = np.random.randint(self.lenB)

        # Read and convert the images from domainA and domainB to RGB arrays
        imageA = np.array(Image.open(self.imagesA[idx_a]).convert("RGB"))
        imageB = np.array(Image.open(self.imagesB[idx_b]).convert("RGB"))

        # Apply transformations to the images if a transformation function is provided
        if self.transforms is not None:
            imageA = self.transforms(imageA)
            imageB = self.transforms(imageB)

        # Return the transformed or original images from domainA and domainB
        return imageA, imageB
