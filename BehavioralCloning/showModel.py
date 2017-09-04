import argparse
#import pydot
#import graphviz

from keras.models import load_model
from keras.utils import plot_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()

    model = load_model(args.model)

    plot_model(model, to_file='model.png')
