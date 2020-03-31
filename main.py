from lib.hdr import hdr
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type = str, help = 'Input directory of images.')
    parser.add_argument('output_dir', type = str, help = 'Output directoru of hdr image and plots.')
    parser.add_argument('-l', type = float, help = 'Lambda factor in Debevec\'s method.', default = 20)
    parser.add_argument('--scale', type = float, help = 'Downscaling factor of image.', default = 1)
    parser.add_argument('--hat', type = str, help = 'Hatting function for pixel values', default = 'linear')
    args = parser.parse_args()
    
    hdr(args.input_dir, args.output_dir, args.hat, args.l, args.scale)