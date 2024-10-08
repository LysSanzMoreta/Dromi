"""
=======================
2023: Lys Sanz Moreta
Dromi: Python package for parallel computation of similarity measures among vector-encoded sequences
=======================
"""
import os,sys,argparse
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import RawTextHelpFormatter
import numpy as np
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository:
     sys.path.insert(1, "{}/dromi/src".format(script_dir))
     import dromi
else:#pip installed module
     import dromi
import dromi.utils as DromiUtils
import dromi.similarities as DromiSimilarities
import dromi.mutual_information as DromiMI

print("Loading dromi module from {}".format(dromi.__file__))
def plot_heatmap(array, title,file_name):
    """Plot heatmap of array
    :param array: Numpy array
    :param title: Plot title
    :param file_name"""
    print("Visualizing heatmap...")
    fig = plt.figure(figsize=(20, 20))
    ax = sns.heatmap(array, cmap='RdYlGn_r',yticklabels=False,xticklabels=False)
    ax.collections[0].set_clim(0, 1)
    plt.title(title,fontsize=20)
    plt.savefig(file_name)
    plt.clf()
    plt.close(fig)

def example_blosum_encoded_sequences(unique_characters=21,random_seqs=False):
    """The similarity computations are performed excluding self similarity. The current position is compared to the other positions in the same site.
    NOTE: I have only implemented similarity matrix with paddings at the end, if requested I might look into paddings with other distributions
    """
    if random_seqs:
        random_result = DromiUtils.SequenceRandomGeneration(["".join(["A"] * 40)] * 300, 50, "no_padding").run()
        seqs, sequences_padded = zip(*random_result)
        max_len = len(max(seqs, key=len))

    else:
        seqs = ["AHPDYRMPIL"] * 1000
        # seqs = ["AHPDYRM",
        #         "AHPHYRM",
        #         "AKPDYRM",
        #         "AHPDYRM",
        #         "AHPDYRM",
        #         "FYRA",
        #         "MRSTVI"]
        # seqs = [
        #         "RGICWMLV",
        #         "RGICWMLV",
        #         "RGVCWMLV",
        #         "RGVCWMLV",
        #         "RGACWMLV",
        #         "RGACFMLV",
        #         "RGLCYMLV",
        #         "RGLCYMLV",
        #         "RGICYMLV",
        #         "RGICYMLV",
        # ]
        max_len = len(max(seqs, key=len))

        padding_result = DromiUtils.SequencePadding(seqs, max_len, method="ends",shuffle=False).run()
        sequences, sequences_padded = zip(*padding_result)  # unpack list of tuples onto 2 lists

    blosum_array, blosum_dict, blosum_array_dict = DromiUtils.create_blosum(unique_characters, "BLOSUM62",
                                                                               zero_characters=["#"],
                                                                               include_zero_characters=True)

    aa_dict = DromiUtils.aminoacid_names_dict(21, zero_characters=["#"])
    sequences_array = np.array(sequences_padded)
    sequences_int = np.vectorize(aa_dict.get)(sequences_array)
    sequences_blosum = np.vectorize(blosum_array_dict.get,signature='()->(n)')(sequences_int)
    sequences_mask = sequences_int.astype(bool)
    storage_folder = "{}".format(script_dir)
    start = time.time()

    results = DromiSimilarities.calculate_similarities_ondisk(sequences_blosum,max_len,sequences_mask,storage_folder,batch_size=300,ksize=3,neighbours=1)
    stop = time.time()
    print("Finished in {}".format(str(datetime.timedelta(seconds=stop-start))))
    plot_heatmap(results.positional_weights,"HEATMAP Positional weights","{}/HEATMAP_positional_weights".format(storage_folder))
    plot_heatmap(results.percent_identity_mean,"HEATMAP Percent Identity mean","{}/HEATMAP_pecent_id_mean".format(storage_folder))
    plot_heatmap(results.cosine_similarity_mean,"HEATMAP Cosine similarity mean","{}/HEATMAP_cosine_similarity_mean".format(storage_folder))
    plot_heatmap(results.kmers_pid_similarity,"HEATMAP Kmers percent identity mean","{}/HEATMAP_kmers_pid_similarity".format(storage_folder))
    plot_heatmap(results.kmers_cosine_similarity_mean,"HEATMAP Kmers cosine similarity mean","{}/HEATMAP_kmers_cosine_similarity_mean".format(storage_folder))

def example_mutual_information():
    """On disk computation of Mutual information between continuous random variables"""

    gene_expression_matrix = np.random.rand(2000,503)
    mi_results = DromiMI.calculate_mutual_information(gene_expression_matrix,bins=5)

    results_dir = ""
    name = ""
    results_dir = "" if not results_dir else f"{results_dir}/"
    name = "" if not name else f"_{name}"
    np.save("{}Mutual_information{}.npy".format(results_dir, name), mi_results["mutual_information"])
    np.save("{}Mutual_information_normalized{}.npy".format(results_dir,name),mi_results["normalized_mutual_information"])

def parse_args(parser):
    parser.add_argument('-analysis', type=str, nargs='?', default="cosine",
                        help='<cosine> \n'
                             '<mutualinfo>')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dromi args",formatter_class=RawTextHelpFormatter)

    args = parse_args(parser)
    if args.analysis == "cosine":
        example_blosum_encoded_sequences()
    elif args.analysis == "mutualinfo":
        example_mutual_information()


