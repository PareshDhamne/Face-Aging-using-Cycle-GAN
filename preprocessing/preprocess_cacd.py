# Import necessary libraries
import os
import shutil
from argparse import ArgumentParser
from scipy.io import loadmat

# Create an ArgumentParser object for parsing command-line arguments
parser = ArgumentParser()
# Define command-line arguments
parser.add_argument('--image_dir',
                    default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/data_image/CACD',
                    help='The CACD200 images dir')
parser.add_argument('--metadata',
                    default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/data_image/CACD.mat',
                    help='The metadata for the CACD2000')
parser.add_argument('--output_dir',
                    default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/data_image',
                    help='The directory to write processed images')

# Define the main function
def main():
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Load metadata from the specified MAT file
    metadata = loadmat(args.metadata)['celebrityImageData'][0][0]
    
    # Extract ages and names from metadata
    ages = [x[0] for x in metadata[0]]
    names = [x[0][0] for x in metadata[-1]]

    # Define age ranges to keep for domainA and domainB
    ages_to_keep_a = [x for x in range(18, 30)]
    ages_to_keep_b = [x for x in range(55, 100)]

    # Initialize lists to store file names for domainA and domainB
    domainA, domainB = [], []
    
    # Populate domainA and domainB based on age ranges
    for age, name in zip(ages, names):
        if age in ages_to_keep_a:
            domainA.append(name)
        if age in ages_to_keep_b:
            domainB.append(name)

    # Determine the minimum length between domainA and domainB
    N = min(len(domainA), len(domainB))
    
    # Trim domainA and domainB to have the same length
    domainA = domainA[:N]
    domainB = domainB[:N]
    
    # Print the number of images in domainA and domainB
    print(f'Images in A {len(domainA)} and B {len(domainB)}')

    # Define output directories for domainA and domainB
    domainA_dir = os.path.join(args.output_dir, 'trainA')
    domainB_dir = os.path.join(args.output_dir, 'trainB')

    # Create output directories if they do not exist
    os.makedirs(domainA_dir, exist_ok=True)
    os.makedirs(domainB_dir, exist_ok=True)

    # Copy images from the source directory to the respective domain directories
    for imageA, imageB in zip(domainA, domainB):
        # Use os.path.join to construct file paths to avoid double backslashes
        shutil.copy(os.path.join(args.image_dir, imageA), os.path.join(domainA_dir, os.path.basename(imageA)))
        shutil.copy(os.path.join(args.image_dir, imageB), os.path.join(domainB_dir, os.path.basename(imageB)))

# Entry point for the script
if __name__ == '__main__':
    # Call the main function if the script is executed
    main()
