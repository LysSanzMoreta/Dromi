import os
import random
import numpy as np
import struct
from typing import Union


def write_3d_array_to_binary(filename, array):
    # Get the shape of the array (x, y, z dimensions)
    x_dim, y_dim, z_dim = array.shape

    # Open the binary file in write mode
    with open(filename, 'wb') as f:
        # Write the dimensions as integers (3 integers)
        f.write(struct.pack('3i', x_dim, y_dim, z_dim))

        # Write the data (flattened array) to the binary file
        array.astype(np.float32).tofile(f)


def get_random_seq(max_seq_len: int, alphabet: str) -> str:
    """Get a random seq with padding.

    :param max_seq_len: max length of sequence
    :type max_seq_len: int
    :param alphabet: the alphabet used
    :type alphabet: str
    :return: A padded sequence
    :rtype: str
    """
    k = random.randint(1, 20)
    s = ''.join(random.choices(alphabet, k=k))
    s = s + (max_seq_len - len(s)) * '#'
    return s


def main(array: np.ndarray, batch_size: Union[int, float], storage_folder: str, file_name_base: str = "data"):
    num_batches = 10
    num_seqs_batch = 6  # tensor 0 dim
    max_seq_len = 20  # tensor 1 dim
    feature_len = 5  # tensor 2 dim

    if array.ndims == 3:
        num_seqs_batch, max_seq_len, feature_len = array.shape
    else:
        num_seqs_batch, max_seq_len = array.shape
        feature_len = 1

    # alphabet = 'ATCG'
    # file_name_base = 'data'
    # folder_name = 'test_data'
    folder_name, dir_name = os.path.basename(storage_folder), os.path.dirname(storage_folder)

    folders(folder_name, storage_folder, overwrite=False)
    # create a fingerprint per char
    # f_dict = {}

    # for char in alphabet + '#':
    #     if char == '#':
    #         f_dict[char] = np.zeros(feature_len)
    #     else:
    #         f_dict[char] = np.random.randn(feature_len)

    # Create the tensors
    for i in range(num_batches):
        # tensor = np.zeros((num_seqs_batch, max_seq_len, feature_len))
        # for j in range(num_seqs_batch):
        #     seq = get_random_seq(max_seq_len, alphabet)
        #     for k, char in enumerate(seq):
        #         tensor[j][k] = f_dict[char]

        # Now save the tensor
        write_3d_array_to_binary(f'{storage_folder}/{file_name_base}_{i}.bin', tensor)
