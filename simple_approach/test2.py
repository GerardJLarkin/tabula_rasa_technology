# import itertools

# # your inputs
# ids1 = ["A", "B", "C"]
# ids2 = ["X", "Y"]
# directions = [f"{i*15}°" for i in range(24)]
# nest_dict = {'f': 0.0, 'v': (0.0, 0.0)}

# # build the nested dict
# nested = {
#     f"{i1}-{i2}": nest_dict
#     for i1, i2 in itertools.product(ids1, ids2)
# }

# # inspect one entry
# key = "A-X"
# print(key, "→", len(nested[key]), "direction‐pairs")
# print(nested)  # first five pairs

def reverse_normalize(value, center, range_min=0, range_max=64):
    """
    Reverses the normalization of a value between -1 and 1 to the original range.

    Args:
        value: The normalized value (-1 to 1).
        center: The center point within the original range.
        range_min: The minimum value of the original range (default 0).
        range_max: The maximum value of the original range (default 64).

    Returns:
        The reversed normalized value in the original range.
    """
    if not -1 <= value <= 1:
        raise ValueError("Normalized value must be between -1 and 1")

    range_width = range_max - range_min
    # Scale the normalized value to the full range width.
    scaled_value = value * (range_width / 2)
    # Shift the scaled value to the correct position relative to the center
    original_value = center + scaled_value

    return original_value


# Example usage:
normalized_value = 0.5  # Example normalized value between -1 and 1
center_point = 20  # Example center point
original_value = reverse_normalize(normalized_value, center_point)
print(f"Reversed normalized value: {original_value}")  # Output will be 42.0
#Example with a negative normalized value
normalized_value = -0.75
original_value = reverse_normalize(normalized_value, center_point)
print(f"Reversed normalized value: {original_value}") #output will be 8.0

#Example within the range limits
normalized_value = 1.0
original_value = reverse_normalize(normalized_value, center_point)
print(f"Reversed normalized value: {original_value}") #output will be 52.0

normalized_value = -1.0
original_value = reverse_normalize(normalized_value, center_point)
print(f"Reversed normalized value: {original_value}") #output will be 12.0
