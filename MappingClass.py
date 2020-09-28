#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
from Bio import SeqIO
import warnings
import sys
import os
warnings.filterwarnings("ignore")


#############################################################################
#############################################################################


def check_files():
    for name, label in dataset_labels.items():
        if os.path.exists(name):
            print("Dataset %s: Found File" % (name))
            run = 1
        else:
            print("Dataset %s: File not exists" % (name))
            run = 0
            break
    return run


def sequence_length():
    dlength = []
    for name, label in dataset_labels.items():
        length = 0
        for seq_record in SeqIO.parse(name, "fasta"):
            seq = seq_record.seq
            if len(seq) > length:
                length = len(seq)
        dlength.append(length)
    # max_length = max(dlength)
    return max(dlength)

        
def file_record(name_seq, mapping, label):
    dataset = open(foutput, 'a')
    dataset.write("%s," % (str(name_seq)))
    for map in mapping:
        dataset.write("%s," % (map))
        # dataset.write("{0:.4f},".format(metric))
    dataset.write(label)
    dataset.write("\n")
    print("Sequence Analyzed!!")
    return


def binary_fourier():
    for finput, label in dataset_labels.items():
        for seq_record in SeqIO.parse(finput, "fasta"):
            seq = seq_record.seq
            seq = seq.upper()
            name_seq = seq_record.name
            list_nucle = ["A", "C", "T", "U", "G"]
            A = []
            C = []
            T = []
            G = []
            for nucle in seq:
                if nucle == "A":
                    A.append(1)
                else:
                    A.append(0)
                if nucle == "C":
                    C.append(1)
                else:
                    C.append(0)
                if nucle == "T" or nucle =="U":
                    T.append(1)
                else:
                    T.append(0)
                if nucle == "G":
                    G.append(1)
                else:
                    G.append(0)
            concat = A + C + T + G
            padding = (max_length - len(A)) * 4
            mapping = np.pad(concat, (0, padding), 'constant')
            file_record(name_seq, mapping, label)
    return


def ref_fourier():
    for finput, label in dataset_labels.items():
        for seq_record in SeqIO.parse(finput, "fasta"):
            seq = seq_record.seq
            seq = seq.upper()
            name_seq = seq_record.name
            mapping = []
            for nucle in seq:
                if nucle == "T" or nucle == "U":
                    mapping.append(0.50)
                elif nucle == "C":
                    mapping.append(0.25)
                elif nucle == "A":
                    mapping.append(1.0)
                elif nucle == "G":
                    mapping.append(0.75)
                else:
                    mapping.append(0.0)
            padding = (max_length - len(mapping))
            mapping = np.pad(mapping, (0, padding), 'constant')
            file_record(name_seq, mapping, label)
    return


#############################################################################    
if __name__ == "__main__":
    print("\n")
    print("###################################################################################")
    print("##########         Feature Extraction: Numerical Mapping Approach       ###########")
    print("##########  Arguments: -i number of datasets -o output -r representation  #########")
    print("##########                 -r:  1 = Binary, 2 = REF                       #########")
    print("##########                 Author: Robson Parmezan Bonidia              ###########")
    print("###################################################################################")
    print("\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', help='Fasta format file | Number of dataset or labels')
    parser.add_argument('-o', '--output', help='Csv format file | E.g., train.csv')
    parser.add_argument('-r', '--representation', help='1 = Binary, 2 = REF')
    args = parser.parse_args()
    n = int(args.number)
    foutput = str(args.output)
    representation = int(args.representation)
    dataset_labels = {}
    for i in range(1, n+1):
        name = input("Dataset %s: " % (i))
        label = input("Label for %s: " % (name))
        print("\n")
        dataset_labels[name] = label
    if check_files() == 1:
        max_length = sequence_length()
        if representation == 1:
            binary_fourier()
        elif representation == 2:
            ref_fourier()
        else:
            print("This package does not contain this representation - Check parameter -r")
    else:
        print("Some file not exists")
#############################################################################
