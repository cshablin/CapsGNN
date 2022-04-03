"""Running CapsGNN."""

from utils import tab_printer, seed_everything
from capsgnn import CapsGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_parser()
    tab_printer(args)
    seed_everything(50)
    model = CapsGNNTrainer(args)
    model.fit()
    #model.score()
    model.test_mse()
    model.save_predictions()

if __name__ == "__main__":
    main()
