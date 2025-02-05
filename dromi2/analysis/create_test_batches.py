"""Function to create test tensors as bin files
"""
import os
import random
import numpy as np
import struct


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
    s = s+(max_seq_len-len(s))*'#'
    return s


def main():
    num_batches = 10
    num_seqs_batch = 6 # tensor 0 dim
    max_seq_len = 20 # tensor 1 dim
    feature_len = 5 # tensor 2 dim
    alphabet = 'ATCG'
    file_name_base = 'data'
    folder_name = 'test_data'


    try:
        os.mkdir(folder_name)
    except:
        pass
    # create a fingerprint per char
    f_dict = {}

    for char in alphabet+'#':
        if char == '#':
            f_dict[char] = np.zeros(feature_len)
        else:
            f_dict[char] = np.random.randn(feature_len)

    # Create the tensors
    for i in range(num_batches):
        tensor = np.zeros((num_seqs_batch, max_seq_len, feature_len))
        for j in range(num_seqs_batch):
            seq = get_random_seq(max_seq_len, alphabet)
            for k, char in enumerate(seq):
                tensor[j][k] = f_dict[char]

        # Now save the tensor
        write_3d_array_to_binary(f'{folder_name}/{file_name_base}_{i}.bin', tensor)



if __name__ == "__main__":
    main()



