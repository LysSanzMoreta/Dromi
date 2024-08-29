"""
=======================
2023: Lys Sanz Moreta
Dromi: Python package for parallel computation of similarity measures among vector-encoded sequences
=======================
"""
import os,sys,argparse
from argparse import RawTextHelpFormatter
from gooey import Gooey, GooeyParser
local_repository=True
script_dir = os.path.dirname(os.path.abspath(__file__))
if local_repository:
     sys.path.insert(1, "{}/dromi/src".format(script_dir))
     import dromi
else:#pip installed module
     import dromi

import Dromi_example as DromiExample
print("Loading dromi module from {}".format(dromi.__file__))


@Gooey(optional_cols=3,program_name="Vegvisir Executable with Pyinstaller",default_size=(1000,1000))
def parse_args():
    #parser = argparse.ArgumentParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    parser = GooeyParser(description="Vegvisir args",formatter_class=RawTextHelpFormatter)
    args = DromiExample.parse_args(parser)
    return args

if __name__ == "__main__":

    args = parse_args()

    if args.analysis == "cosine":
        DromiExample.example_blosum_encoded_sequences()
    elif args.analysis == "mutualinfo":
        DromiExample.example_mutual_information()