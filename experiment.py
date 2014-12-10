import argparse
import utilities.configutils as cfgutils
from  experiment.base import Experiment
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                choices=['imbd', 'sraa', '20news'],
                help='training data (libSVM format)')

ap.add_argument('--verbose',
                action='store_true',
                help='to print progress of experiment')


ap.add_argument('--config',
                metavar='CONFIG_FILE',
                type=str,
                default='./default.cfg',
                help='Experiment configuration file')

def main():
    args = ap.parse_args()
    print args.train
    config = cfgutils.get_config(args.config)
    experiment = Experiment(args.train, config, verbose=args.verbose)
    experiment.start()


if __name__ == "__main__":
    main()