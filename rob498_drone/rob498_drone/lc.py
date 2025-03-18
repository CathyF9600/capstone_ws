from collections import defaultdict

def has_same_unique_integers(nums1, nums2, k):
    def get_unique_counts(nums, k):
        freq = defaultdict(int)  # Frequency dictionary
        unique_nums = defaultdict(int)  # Tracks presence of unique numbers
        n = len(nums)
        result = []  # Stores unique integers for each window

        # Initialize the first window
        for i in range(k):
            num = nums[i]
            freq[num] += 1
            if freq[num] == 1:
                unique_nums[num] = 1  # Add to unique list
            elif freq[num] == 2:
                del unique_nums[num]  # No longer unique

        result.append(tuple(unique_nums.keys()))  # Store unique numbers

        # Slide the window
        for i in range(k, n):
            # Remove the left element
            left = nums[i - k]
            freq[left] -= 1
            if freq[left] == 0:
                del freq[left]
            if freq[left] == 1:
                unique_nums[left] = 1  # Becomes unique again
            elif left in unique_nums:
                del unique_nums[left]  # No longer unique

            # Add the right element
            right = nums[i]
            freq[right] += 1
            if freq[right] == 1:
                unique_nums[right] = 1  # Becomes unique
            elif freq[right] == 2:
                del unique_nums[right]  # No longer unique

            result.append(tuple(unique_nums.keys()))  # Store unique numbers

        return result

    # Get unique integer sets for each window in both arrays
    unique_nums1 = get_unique_counts(nums1, k)
    unique_nums2 = get_unique_counts(nums2, k)
    print(unique_nums1, unique_nums2)
    # Check if any window has the same unique numbers
    return any(window in unique_nums2 for window in unique_nums1)


nums1 = [1, 2, 3, 2, 4]
nums2 = [4, 5, 6, 5, 1]
k = 3
nums1 = [1,3,2,2,1,3]
nums2 = [1,2,3]

print(has_same_unique_integers(nums1, nums2, k)) 