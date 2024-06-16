# Import necessary libraries
import os
import shutil
from argparse import ArgumentParser

# Create an ArgumentParser object to handle command line arguments
parser = ArgumentParser()
# Define command line arguments for data directory and output directory
parser.add_argument('--data_dir',
                    default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/data_img/UTKFace',
                    help='The UTKFace aligned images dir')
parser.add_argument('--output_dir',
                    default='D:/FaceAgingCAAE/FaceAgingFF/Fast-AgingGAN/data_img',
                    help='The directory to write processed images')

# Define the main function
def main():
    # Parse command line arguments
    args = parser.parse_args()

    # Get a list of image names in the specified data directory with the '.jpg' extension
    image_names = [x for x in os.listdir(args.data_dir) if x.endswith('.jpg')]
    # Print the total number of images found
    print(f"Total images found: {len(image_names)}")

    # Extract ages from image names by splitting on '_' and taking the first part
    ages = [int(x.split('_')[0]) for x in image_names]

    # Define age ranges to keep for domain A and domain B
    ages_to_keep_a = [x for x in range(18, 29)]
    ages_to_keep_b = [x for x in range(40, 120)]

    # Initialize lists to store image names for domain A and domain B
    domainA, domainB = [], []

    # Iterate over image names and ages to separate them into domain A and domain B
    for image_name, age in zip(image_names, ages):
        if age in ages_to_keep_a:
            domainA.append(image_name)
        elif age in ages_to_keep_b:
            domainB.append(image_name)

    # Find the minimum number of images between domain A and domain B
    N = min(len(domainA), len(domainB))
    # Limit the number of images in each domain to N
    domainA = domainA[:N]
    domainB = domainB[:N]

    # Print the number of images in domain A and domain B
    print(f"Image in A: {len(domainA)} and B: {len(domainB)}")

    # Create directories for domain A and domain B in the specified output directory
    domainA_dir = os.path.join(args.output_dir, 'trainA')
    domainB_dir = os.path.join(args.output_dir, 'trainB')
    os.makedirs(domainA_dir, exist_ok=True)
    os.makedirs(domainB_dir, exist_ok=True)

    # Copy images from the data directory to the corresponding domain directories
    for imageA, imageB in zip(domainA, domainB):
        shutil.copy(os.path.join(args.data_dir, imageA), os.path.join(domainA_dir, imageA))
        shutil.copy(os.path.join(args.data_dir, imageB), os.path.join(domainB_dir, imageB))

# Execute the main function if the script is run as the main program
if __name__ == '__main__':
    main()