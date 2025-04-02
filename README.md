# Face-Aging-using-Cycle-GAN

## Project Overview

  This project leverages Generative Adversarial Networks (GANs) to create realistic simulations of human facial aging. By transforming input images, it can depict age progression while retaining the subject’s unique features. This technology is valuable in fields like entertainment, forensic analysis, cosmetic research, and scientific studies.

## Business Objective

  - The primary goal is to realistically simulate facial aging while preserving individual characteristics:

  - **Maximize:** The realism and accuracy of aging simulations to reflect true facial changes over time.

  - **Minimize:** Distortion or loss of facial features, ensuring each transformation remains true to the person’s identity.
 
## Technologies Used

  - This project utilizes a combination of machine learning and computer vision libraries to achieve its objectives:

  -  **Python:** Core programming language for flexibility and extensive ML ecosystem.

  -  **TensorFlow:** Provides deep learning capabilities for training GANs efficiently.

  -  **SciPy:** Used for scientific computing and numerical calculations.

  -  **OpenCV:** Handles image processing, transformations, and augmentations.

  -  **NumPy:** Offers fast array operations and numerical functions.

  -  **Pillow:** Image handling and manipulation.

## Impact of the Project

This project can:

  -  **Entertainment:** Create age-progressed characters for movies, games, and virtual reality.

  -  **Forensics:** Assist in identifying missing persons and criminals over time.

  -  **Cosmetic Insights:** Help study aging effects and predict future facial changes.

  -  **Scientific Research:** Contribute to understanding facial aging patterns and genetics.

## Features

- Realistic Aging Simulation: Generates age-progressed or rejuvenated facial images while preserving individual identity.

- Bidirectional Transformation: Supports both aging and de-aging transformations through GAN models.

- High-Quality Image Generation: Uses residual blocks in the Generator to maintain image quality and detail.

- Efficient Training Pipeline: Optimizes GAN training with configurable architectures and loss functions.

- Data Preprocessing: Utilizes the CACD dataset with augmentation and normalization to enhance model robustness.

## Pretrained Model Inference
To apply the pretrained model to your images, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the AgingGAN model using:
   ```
   python weights_main.py --config configs/aging_gan.yaml --weights_dir saved_weights
   ```
5. Measure performance with:
   ```
   python timing.py
   ```
7. Run inference on new images using:
   ```
   python infer.py
   ```
## File Descriptions

-  **gan_module.py**

      Defines the AgingGAN model structure, including Generator and Discriminator networks. Implements adversarial and cycle consistency losses to train the GAN effectively. Supports forward and reverse aging   transformations.

-  **models.py**

      Contains neural network architectures for the Generator and Discriminator. The Generator leverages residual blocks to capture fine details, while the Discriminator evaluates image realism.

-  **timing.py**

      Measures the average time taken for model inference. Conducts 50 iterations with the first 10 as a warm-up to ensure accurate timing results.

-  **weights_main.py**

      Loads configurations, sets up the AgingGAN, trains the model, and saves weights. Generates model architecture diagrams using torchviz.

-  **infer.py**

      Performs aging transformations on input images using trained weights. Supports both aging and rejuvenation scenarios.

-  **dataset.py**

      Handles data loading and preprocessing using the CACD dataset. Supports augmentation techniques and normalization to improve model robustness.
