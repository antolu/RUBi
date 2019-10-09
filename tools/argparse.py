from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # Arguments concerning the environment of the repository
    parser.add_argument("-d", "--dataset", type=str, default="vqa-v2-cp",
                        help="The dataset to train/test on.")
    parser.add_argument("--datadir", type=str, default="datasets",
                        help="The directory containing the datasets.")

    # Arguments concerning training and testing the model
    traintest = parser.add_mutually_exclusive_group(required=True)
    traintest.add_argument("--train", action="store_true",
                           help="Train the model")
    traintest.add_argument("--test", action="store_true",
                           help="Test the model")
    parser.add_argument("--no-epochs", type=int, default=1000000,
                        help="Number of epochs to train the model")
    parser.add_argument("-lr", "--lr", type=float, default=1.5e-4,
                        help="The learning rate for the Adam optimiser.")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="The batchsize to use in training")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="The difference between losses between iterations \
                        to break.")
    parser.add_argument("-l", "--loss-weights", nargs=2, type=float,
                        default=(1, 1), help="The weights which determine the \
                        importance of the RUBi and question-only loss. ")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 or FP32 in training.")
    parser.add_argument("--rubi", action="store_true",
                        help="Use RUBi question-only branch during training.")

    # Other arguments
    parser.add_argument("Baseline", type=str, help="Valid baseline net: rubi, san, updn.")

    args = parser.parse_args()

    return args
