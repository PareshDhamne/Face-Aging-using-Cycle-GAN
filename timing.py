# Import the 'default_timer' function from the 'timeit' module as 'timer'.
from timeit import default_timer as timer

# Import the 'torch' library.
import torch

# Define a function named 'time_model' that takes a 'model' and 'input_size' as arguments.
def time_model(model, input_size):
    # Set the model to evaluation mode.
    model.eval()
    
    # Initialize variables 'count' and 'duration' to 0.
    count, duration = 0, 0
    
    # Loop 50 times.
    for i in range(50):
        # Record the starting time.
        start = timer()
        
        # Generate random input data using torch.rand and pass it through the model.
        _ = model(torch.rand(size=input_size))
        
        # If the iteration index 'i' is less than 10, skip the rest of the loop and continue.
        if i < 10:
            continue
        
        # Record the duration of the model inference (excluding the warm-up phase).
        duration += timer() - start
        # Increment the count variable.
        count += 1

    # Calculate and return the average duration per inference.
    return duration / count

# Define the main function.
def main():
    # Import the 'Generator' class from the 'models' module.
    from models import Generator
    
    # Create an instance of the 'Generator' class with specified parameters.
    model = Generator(32, 9)
    
    # Measure the time taken for model inference using the 'time_model' function.
    duration = time_model(model, [1, 3, 512, 512])
    
    # Print the result.
    print("Time Taken (excluding warmup): ", duration)

# If the script is executed as the main module, run the 'main' function.
if __name__ == '__main__':
    main()