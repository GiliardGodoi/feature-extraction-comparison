import argparse
import pandas as pd
import re
import os
from os import path

def _add_tuple(aa, bb):
    return (aa[0] + bb[0], aa[1])

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, help='folder to search for  the input files')
parser.add_argument('--start', type=str, help='starts whit (prefix)')
parser.add_argument('--out', type=str, help='where to put')

args = parser.parse_args()

print('Looking in the folder: ', args.folder)
print('Prefix: ', args.start)
print('Output files: ', args.out)

prefix = args.start

files = [f for f in os.listdir(args.folder) if f.startswith(prefix) ]

pattern = re.compile(r'l(\d)')

def _key(filename):
    result = pattern.findall(filename)
    if result:
        number = result[0]
        return int(number)
    else:
        return 0

files = sorted(files,key=_key)

print("Concatenate files: ")
for f in files : print('\t', f)

data = list()
n_recors = 0
n_columns = 0
for f in files :
    df = pd.read_csv(path.join(args.folder, f), header=0)
    r, c = df.shape
    n_recors += r
    n_columns = c
    data.append(df)

frame = pd.concat(data, ignore_index=True)

assert frame.shape == (n_recors, n_columns)

frame.to_csv(args.out, header=True, index=False)
