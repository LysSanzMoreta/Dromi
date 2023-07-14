"""
=======================
2023: Lys Sanz Moreta
Dromi: Python package for parallel computation of similarity measures among vector-encoded sequences
=======================
"""
import random
import numpy as np
from collections import defaultdict
import warnings
import Bio.Align
def aminoacid_names_dict(aa_types,zero_characters = []):
    """ Returns an aminoacid associated to a integer value
    All of these values are mapped to 0:
        # means empty value/padding
        - means gap in an alignment
        * means stop codon
    :param int aa_types: amino acid probabilities, this number correlates to the number of different aa types in the input alignment
    :param list zero_characters: character(s) to be set to 0
    """
    if aa_types == 20 :
        assert len(zero_characters) == 0, "No zero characters allowed, please set zero_characters to empty list"
        aminoacid_names = {"R":0,"H":1,"K":2,"D":3,"E":4,"S":5,"T":6,"N":7,"Q":8,"C":9,"G":10,"P":11,"A":12,"V":13,"I":14,"L":15,"M":16,"F":17,"Y":18,"W":19}
    elif aa_types == 21:
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20}
    else :
        aminoacid_names = {"R":1,"H":2,"K":3,"D":4,"E":5,"S":6,"T":7,"N":8,"Q":9,"C":10,"G":11,"P":12,"A":13,"V":14,"I":15,"L":16,"M":17,"F":18,"Y":19,"W":20,"B":21,"Z":22,"X":23}
    if zero_characters:
        for element in zero_characters:
                aminoacid_names[element] = 0
    aminoacid_names = {k: v for k, v in sorted(aminoacid_names.items(), key=lambda item: item[1])} #sort dict by values (for dicts it is an overkill, but I like ordered stuff)
    return aminoacid_names

def create_blosum(aa_types,subs_matrix_name,zero_characters=[],include_zero_characters=False):
    """
    Builds an array containing the blosum scores per character
    :param aa_types: amino acid probabilities, determines the choice of BLOSUM matrix
    :param str subs_matrix_name: name of the substitution matrix, check availability at /home/lys/anaconda3/pkgs/biopython-1.76-py37h516909a_0/lib/python3.7/site-packages/Bio/Align/substitution_matrices/data
    :param bool include_zero_characters : If True the score for the zero characters is kept in the blosum encoding for each amino acid, so the vector will have size 21 instead of just 20
    """

    if aa_types > 21 and not subs_matrix_name.startswith("PAM"):
        warnings.warn("Your dataset contains special amino acids. Switching your substitution matrix to PAM70")
        subs_matrix_name = "PAM70"
    elif aa_types == 20 and len(zero_characters) !=0:
        raise ValueError("No zero characters allowed, please set zero_characters to empty list")

    subs_matrix = Bio.Align.substitution_matrices.load(subs_matrix_name)
    aa_list = list(aminoacid_names_dict(aa_types,zero_characters=zero_characters).keys())

    if zero_characters:
        index_gap = aa_list.index("#")
        aa_list[index_gap] = "*" #in the blosum matrix gaps are represented as *

    subs_dict = defaultdict()
    subs_array = np.zeros((len(aa_list) , len(aa_list) ))
    for i, aa_1 in enumerate(aa_list):
        for j, aa_2 in enumerate(aa_list):
            if aa_1 != "*" and aa_2 != "*":
                subs_dict[(aa_1,aa_2)] = subs_matrix[(aa_1, aa_2)]
                subs_dict[(aa_2, aa_1)] = subs_matrix[(aa_1, aa_2)]
            else:
                subs_dict[(aa_1, aa_2)] = -1 #gap penalty

            subs_array[i, j] = subs_matrix[(aa_1, aa_2)]
            subs_array[j, i] = subs_matrix[(aa_2, aa_1)]

    names = np.concatenate((np.array([float("-inf")]), np.arange(0,len(aa_list))))
    subs_array = np.c_[ np.arange(0,len(aa_list)), subs_array ]
    subs_array = np.concatenate((names[None,:],subs_array),axis=0)

    #subs_array[1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]
    #subs_array[:,1] = np.zeros(aa_types+1)  #replace the gap scores for zeroes , instead of [-4,-4,-4...]

    #blosum_array_dict = dict(enumerate(subs_array[1:,2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements
    if include_zero_characters or not zero_characters:
        blosum_array_dict = dict(enumerate(subs_array[1:,1:]))
    else:
        blosum_array_dict = dict(enumerate(subs_array[1:, 2:])) # Highlight: Changed to [1:,2:] instead of [1:,1:] to skip the scores for non-aa elements

    #blosum_array_dict[0] = np.full((aa_types),0)  #np.nan == np.nan is False ...

    return subs_array, subs_dict, blosum_array_dict
class SequencePadding(object):
    """Performs padding of a list of given sequences to a given len"""
    def __init__(self,sequences,seq_max_len,method,shuffle):
        self.sequences = sequences
        self.seq_max_len = seq_max_len
        self.method = method
        self.shuffle = shuffle
        self.random_seeds = list(range(len(sequences)))

    def run(self):

        if self.method == "no_padding":
            result = list(map(lambda seq,seed: self.no_padding(seq,seed, self.seq_max_len,self.shuffle),self.sequences,self.random_seeds))
        elif self.method == "ends":
            result = list(map(lambda seq,seed: self.ends_padding(seq,seed, self.seq_max_len,self.shuffle),self.sequences,self.random_seeds))
        elif self.method == "random":
            result = list(map(lambda seq,seed: self.random_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds))
        elif self.method == "borders":
            result = list(map(lambda seq,seed: self.border_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds))
        elif self.method == "replicated_borders":
            result = list(map(lambda seq,seed: self.replicated_border_padding(seq,seed, self.seq_max_len,self.shuffle), self.sequences,self.random_seeds))
        else:
            raise ValueError(
                "Padding method <{}> not implemented, please choose among <no_padding,ends,random,borders,replicated_borders>".format(
                    self.method))

        return result

    def no_padding(self,seq,seed,max_len,shuffle):
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        return (list(seq),list(seq))

    def ends_padding(self,seq,seed,max_len,shuffle):
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        return (list(seq.ljust(max_len, "#")),list(seq.ljust(max_len, "#")))

    def random_padding(self,seq,seed, max_len,shuffle):
        """Randomly pad sequence. Introduces <n pads> in random places until max_len"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            idx = np.array(random.sample(range(0, max_len), pad), dtype=int)
            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def border_padding(self,seq,seed, max_len,shuffle):
        """For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:#same amount of padding added at the beginning and the end of the sequence
                idx_pads = np.concatenate(
                    [np.arange(0, int(half_pad)), np.arange(max_len - int(half_pad), max_len)])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                idx_pads_dict = {0: np.concatenate([np.arange(0, int(half_pad) + 1), np.arange(max_len - int(half_pad), max_len)]),
                                 1: np.concatenate([np.arange(0, int(half_pad)),np.arange(max_len - (int(half_pad) + 1), max_len)])}
                idx_pads = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def replicated_border_padding(self, seq,seed, max_len,shuffle):
        """
        Inspired by "replicated" padding in Convolutional NN https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        if shuffle:
            random.seed(seed)
            seq = "".join(random.sample(list(seq), len(seq)))
        #random.seed(91)
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:  # same amount of paddng added at the beginning and the end of the sequence
                start = np.arange(0, int(half_pad))
                end = np.arange(max_len - int(half_pad), max_len)
                idx_pads = np.concatenate(
                    [start, end])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),
                                      dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                start_0 = np.arange(0, int(half_pad) + 1)
                end_0 = np.arange(max_len - int(half_pad), max_len)
                start_1 = np.arange(0, int(half_pad))
                end_1 = np.arange(max_len - (int(half_pad) + 1), max_len)
                idx_pads_dict = {
                    0: [np.concatenate([start_0, end_0]),start_0,end_0],
                    1: [np.concatenate([start_1, end_1]),start_1,end_0]}
                idx_pads,start,end = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            new_seq_mask = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            new_seq_mask[mask] = np.array(seq)
            if start.size != 0 and end.size == 0:
                new_seq[~mask] = np.array(seq[:len(start)])
            elif start.size == 0 and end.size != 0:
                    new_seq[~mask] = np.array(seq[-len(end):])
            else:
                new_seq[~mask] = np.concatenate([np.array(seq[:len(start)]),np.array(seq[-len(end):])])
            return (new_seq.tolist(), new_seq_mask.tolist())
        else:
            return (seq,seq)
class SequenceRandomGeneration(object):
    """Generates random sequences given a sequence length"""
    def __init__(self,sequences,seq_max_len,method):
        self.sequences = sequences
        self.seq_max_len = seq_max_len
        self.method = method
        self.random_seeds = list(range(len(sequences)))
        self.aminoacids_list = np.array(list(aminoacid_names_dict(20).keys()))

    def run(self):

        # padded_sequences = { "no_padding": list(map(lambda seq,seed: self.no_padding(seq,seed, self.seq_max_len),self.sequences,self.random_seeds)),
        #                     "ends": list(map(lambda seq,seed: self.ends_padding(seq,seed, self.seq_max_len),self.sequences,self.random_seeds)),
        #                     "random":list(map(lambda seq,seed: self.random_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds)),
        #                     "borders":list(map(lambda seq,seed: self.border_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds)),
        #                     "replicated_borders":list(map(lambda seq,seed: self.replicated_border_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds))
        #                     }

        if self.method == "no_padding":
            result = list(map(lambda seq,seed: self.no_padding(seq,seed, self.seq_max_len),self.sequences,self.random_seeds))
        elif self.method == "ends":
            result = list(map(lambda seq,seed: self.ends_padding(seq,seed, self.seq_max_len),self.sequences,self.random_seeds))
        elif self.method == "random":
            result = list(map(lambda seq,seed: self.random_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds))
        elif self.method == "borders":
            result = list(map(lambda seq,seed: self.border_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds))
        elif self.method == "replicated_borders":
            result = list(map(lambda seq,seed: self.replicated_border_padding(seq,seed, self.seq_max_len), self.sequences,self.random_seeds))
        else:
            raise ValueError(
                "Padding method <{}> not implemented, please choose among <no_padding,ends,random,borders,replicated_borders>".format(
                    self.method))

        return result

    def no_padding(self,seq,seed,max_len):
        np.random.seed(seed)
        seq = self.aminoacids_list[np.random.choice(len(self.aminoacids_list),len(seq))]
        return (list(seq),list(seq))


    def ends_padding(self,seq,seed,max_len):
        np.random.seed(seed)
        seq = self.aminoacids_list[np.random.choice(len(self.aminoacids_list), len(seq))]
        seq = "".join(seq)
        return (list(seq.ljust(max_len, "#")),list(seq.ljust(max_len, "#")))

    def random_padding(self,seq,seed, max_len):
        """Randomly pad sequence. Introduces <n pads> in random places until max_len"""
        np.random.seed(seed)
        seq = self.aminoacids_list[np.random.choice(len(self.aminoacids_list), len(seq))]
        #seq = "".join(seq)
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            idx = np.array(random.sample(range(0, max_len), pad), dtype=int)
            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def border_padding(self,seq,seed, max_len):
        """For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        np.random.seed(seed)
        seq = self.aminoacids_list[np.random.choice(len(self.aminoacids_list), len(seq))]
        #seq = "".join(seq)
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:#same amount of padding added at the beginning and the end of the sequence
                idx_pads = np.concatenate(
                    [np.arange(0, int(half_pad)), np.arange(max_len - int(half_pad), max_len)])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                idx_pads_dict = {0: np.concatenate([np.arange(0, int(half_pad) + 1), np.arange(max_len - int(half_pad), max_len)]),
                                 1: np.concatenate([np.arange(0, int(half_pad)),np.arange(max_len - (int(half_pad) + 1), max_len)])}
                idx_pads = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            return (new_seq.tolist(),new_seq.tolist())
        else:
            return (seq,seq)

    def replicated_border_padding(self, seq,seed, max_len):
        """
        Inspired by "replicated" padding in Convolutional NN https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        For sequences shorter than seq_max_len introduced padding in the beginning and the ends of the sequences.
        If the amount of padding needed is divisible by 2 then the padding is shared evenly at the bginning and the end of the sequence.
        Otherwise randomly, the beginning or the end of the sequence will receive more padding"""
        np.random.seed(seed)
        seq = self.aminoacids_list[np.random.choice(len(self.aminoacids_list), len(seq))]
        #seq = "".join(seq)
        pad = max_len - len(seq)
        seq = list(seq)
        if pad != 0:
            half_pad = pad / 2
            even_pad = [True if pad % 2 == 0 else False][0]
            if even_pad:  # same amount of paddng added at the beginning and the end of the sequence
                start = np.arange(0, int(half_pad))
                end = np.arange(max_len - int(half_pad), max_len)
                idx_pads = np.concatenate(
                    [start, end])
            else:
                idx_choice = np.array(random.sample(range(0, 1), 1),
                                      dtype=int).item()  # random choice of adding the extra padding to the beginning or end
                start_0 = np.arange(0, int(half_pad) + 1)
                end_0 = np.arange(max_len - int(half_pad), max_len)
                start_1 = np.arange(0, int(half_pad))
                end_1 = np.arange(max_len - (int(half_pad) + 1), max_len)
                idx_pads_dict = {
                    0: [np.concatenate([start_0, end_0]),start_0,end_0],
                    1: [np.concatenate([start_1, end_1]),start_1,end_0]}
                idx_pads,start,end = idx_pads_dict[idx_choice]

            new_seq = np.array(["#"] * max_len)
            new_seq_mask = np.array(["#"] * max_len)
            mask = np.full(max_len, True)
            mask[idx_pads] = False
            new_seq[mask] = np.array(seq)
            new_seq_mask[mask] = np.array(seq)
            if start.size != 0 and end.size == 0:
                new_seq[~mask] = np.array(seq[:len(start)])
            elif start.size == 0 and end.size != 0:
                    new_seq[~mask] = np.array(seq[-len(end):])
            else:
                new_seq[~mask] = np.concatenate([np.array(seq[:len(start)]),np.array(seq[-len(end):])])
            return (new_seq.tolist(), new_seq_mask.tolist())
        else:
            return (seq,seq)
