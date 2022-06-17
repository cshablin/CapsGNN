"""Running CapsGNN."""

from utils import tab_printer
from capsgnn import CapsGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_parser()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    model.fit()
    model.score()
    model.save_predictions()


# my input arguments for this script
# --train-graph-folder ...\CapsGNN_fork\input\train_\ --test-graph-folder ...\CapsGNN_fork\input\test_\ --prediction-path ...\CapsGNN_fork\temp_output\watts_predictions.csv --epochs 50 --batch-size 128 --gcn-layers 4 --capsule-dimensions 8 --number-of-capsules 8 --lambd 0.5 --learning-rate 0.001 --weight-decay 0.0001
if __name__ == "__main__":
    main()
