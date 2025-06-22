# list merge function
# function merge lists with common elements
def merge_lists_with_common_elements(nested_lists):
    result = []  # To store merged lists

    for sublist in nested_lists:
        # Check if this sublist overlaps with any list in the result
        for merged_list in result:
            if set(sublist) & set(merged_list):  # Common elements exist
                merged_list.extend(sublist)  # Add all elements
                merged_list[:] = list(set(merged_list))  # Remove duplicates
                break
        else:
            # If no overlap found, add the sublist as a new group
            result.append(sublist[:])

    return result