import argparse
import pandas as pd

def _add_tuple(aa, bb):
    return (aa[0] + bb[0], aa[1])

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, help='files to concatenate', nargs='+')
parser.add_argument('--out', type=str, help='where to put')

args = parser.parse_args()

print(args.files)
print('\n...\n')
print(args.out)

frame1 = pd.read_csv(args.files[0], header=0)
frame2 = pd.read_csv(args.files[1], header=0)

frame3 = pd.concat([frame1, frame2], ignore_index=True)

assert frame3.shape == _add_tuple(frame1.shape, frame2.shape)

frame3.to_csv(args.out, header=True, index=False)
