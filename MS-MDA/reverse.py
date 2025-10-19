def reverse_array(arr):
    return arr[::-1]

# 示例用法
batch_sizes = [16, 32, 64, 128, 256]
CSESN_S3_means = [88.02, 89.03, 90.83, 91.2, 91.33]
CSESN_S3_std = [5, 4, 4, 3, 3]

CSUB_S3_means = [90.67, 86.56, 89.22, 86.55, 84.02]
CSUB_S3_std = [6, 5, 5, 4, 4]

CSESN_S4_means = [64.11, 65.7, 65.7, 73.27, 75.2]
CSESN_S4_std = [4, 4, 3, 3, 2]

CSUB_S4_means = [62.60, 70.85, 58.34, 62.6, 63.45]
CSUB_S4_std = [5, 5, 4, 4, 3]


reversed_array = reverse_array(CSESN_S3_means)
print(reversed_array)
reversed_array = reverse_array(CSUB_S3_means)
print(reversed_array)
reversed_array = reverse_array(CSESN_S4_means)
print(reversed_array)
reversed_array = reverse_array(CSUB_S4_means)
print(reversed_array)