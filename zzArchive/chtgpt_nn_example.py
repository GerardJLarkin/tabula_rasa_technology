import numpy as np

def compare_pixel_to_neighbors(data, frame, i, j):
    """
    Compare the pixel at position (frame, i, j) to its 26 neighbors
    in a 3D data array (frames x height x width).

    :param data: 3D NumPy array of shape (frames, height, width)
    :param frame: Index of the frame (depth) to compare
    :param i: Row index of the pixel in the frame
    :param j: Column index of the pixel in the frame
    :return: 26 neighboring pixel values and the central pixel value
    """
    # Define the neighborhood range (we are assuming no boundary issues for simplicity)
    # We will handle the boundary check separately if needed
    
    neighbors = data[frame-1:frame+2, i-1:i+2, j-1:j+2]  # Extract the 3x3x3 neighborhood
    
    center_pixel = data[frame, i, j]  # Central pixel value
    
    # Remove the central pixel from the neighbors array for comparison
    neighbors_flat = neighbors.flatten()
    neighbors_flat = np.delete(neighbors_flat, 13)  # Remove the central pixel
    
    # Compare the center pixel to the neighbors
    comparison_result = center_pixel > neighbors_flat  # Example: compare if center is greater
    
    return comparison_result, neighbors_flat, center_pixel

# Example 3D data array (frames x height x width)
data = np.random.randint(0, 256, (30, 720, 1280))  # 10 frames of 5x5 images

# Select a pixel to compare at (frame 5, row 2, column 2)
result, neighbors, center = compare_pixel_to_neighbors(data, 5, 2, 2)

print("Central pixel:", center)
print("Neighboring pixels:", neighbors)
print("Comparison result (center > neighbors):", result)
