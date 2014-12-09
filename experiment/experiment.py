# experiment.py
import argparse
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                choices=['imbd', 'sraa', '20news']
                help='training data (libSVM format)')


ap.add_argument('--config',
                metavar='CONFIG_FILE',
                type=str,
                default='./default.cfg',
                help='Experiment configuration file')

def main():
	args = ap.parse_args()

	config = load_config(args.train)
	experiment = Experiment(args.train, config)
	experiment.start()


if __name__ == "main":
	main()