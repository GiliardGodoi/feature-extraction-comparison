#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import collections
import argparse
import math 
from Bio import SeqIO
from itertools import product


def header(ksize):
    file = open(foutput, 'a')
    file.write("%s," % ("nameseq"))
    for i in range(1, ksize+1):
        file.write("k" + str(i) + ",")
    file.write("class")
    file.write("\n")
    return


def chunks(seq, win, step):
    seqlen = len(seq)
    for i in range(0,seqlen,step):
        j = seqlen if i+win>seqlen else i+win
        yield seq[i:j]
        if j==seqlen: break
    return        
    

def chunksTwo(seq, win):
    seqlen = len(seq)
    for i in range(seqlen):
        j = seqlen if i+win>seqlen else i+win
        yield seq[i:j]
        if j==seqlen: break
    return

            
def fileRecord(name_seq):
    file = open(foutput, 'a')
    file.write("%s," % (str(name_seq)))
    for data in informationEntropy:
        file.write("%s," % (str(data)))
    file.write(labelDataset)
    file.write("\n")
    print ("Recorded Sequence!!!")
    return
    

def findKmers():
    header(ksize)
    global informationEntropy
    for seq_record in SeqIO.parse(finput, "fasta"):
        seq = seq_record.seq
        seq = seq.upper()
        name_seq = seq_record.name
        informationEntropy = []
        for k in range(1, ksize+1):
            probabilities = []
            kmer = {}
            totalWindows = (len(seq) - k) + 1 # (L - k + 1)
            for subseq in chunksTwo(seq, k):
                if subseq in kmer:
                    # print(subseq)
                    kmer[subseq] = kmer[subseq] + 1
                else:
                    kmer[subseq] = 1
            for key, value in kmer.items():
                # print (key)
                # print (value)
                probabilities.append(value/totalWindows)
            entropyEquation = [(p * math.log(p, 2)) for p in probabilities]
            entropy = -(sum(entropyEquation))
            informationEntropy.append(entropy)
        fileRecord(name_seq)
    return

        
#############################################################################    
if __name__ == "__main__":
    print("\n")
    print("###################################################################################")
    print("######################## Feature Extraction: k-mer scheme #########################")
    print("##########   Arguments: python3.5 -i input -o output -l label -k kmer   ###########")
    print("##########               Author: Robson Parmezan Bonidia                ###########")
    print("###################################################################################")
    print("\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Fasta format file, E.g., test.fasta')
    parser.add_argument('-o', '--output', help='CSV format file, E.g., test.csv')
    parser.add_argument('-l', '--label', help='Dataset Label, E.g., lncRNA, mRNA, sncRNA ...')
    parser.add_argument('-k', '--kmer', help='Range of k-mer, E.g., 1-mer (1) or 2-mer (1, 2) ...')
    args = parser.parse_args()
    finput = str(args.input)
    foutput = str(args.output)
    labelDataset = str(args.label)
    ksize = int(args.kmer)
    stepw = 1
    findKmers()
#############################################################################