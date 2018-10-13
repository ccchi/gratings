from grating_tools import *
import argparse

parser=argparse.ArgumentParser(description='Extract 1 dimensional data from line pattern x-ray scattering dataset.')
parser.add_argument('output',metavar='output',type=str,help='File name for output .txt file.')
parser.add_argument('prefix',metavar='prefix',type=str,help='File path and prefix for image names. e.g. "/home/Documents/image". File format assumed to be .fits')

args=parser.parse_args()
spect=reduce_dataset(args.prefix)
np.savetxt(args.output,spect)
