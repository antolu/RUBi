from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    # Arguments concerning the environment of the repository
    parser.add_argument("-d", "--dataset", type=str, default="vqa-v2-cp",
                        help="The dataset to train/test on.")
    parser.add_argument("--datadir", type=str, default="data",
                        help="Path to the root directory containing the datasets.")
    parser.add_argument('--dir_images', type=str, default='images',
                        help="Path to the root directory containing the image datasets")
    parser.add_argument('--dir_data', default='data', type=str,
                        help="Path to the root directory containing the other data")
    parser.add_argument('--dir_model', default='Models/',
                        help='Path to project data')

    parser.add_argument('--answer-type', default='number', type=str, dest="answer_type", choices=["number", "yes-no", "other"], help='answer type (number | yes-no | other)')


    # Arguments concerning training and testing the model
    traintest = parser.add_mutually_exclusive_group(required=True)
    traintest.add_argument("--train", action="store_true",
                           help="Train the model")
    traintest.add_argument("--test", action="store_true",
                           help="Test the model")
    parser.add_argument("--no-epochs", type=int, default=1000000, dest="no_epochs",
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
    parser.add_argument("--opt-level", dest="opt_level", type=str, default="O1",
                        help="Which optimisation to use for mixed precision training.")
    parser.add_argument("--rubi", action="store_true",
                        help="Use RUBi question-only branch during training.")

    parser.add_argument('--workers', default=8, type=int,
                        help="Number of workers for training the network")
    parser.add_argument('--seed', default=123, type=int,
                        help="The random seed for VQA-CP")
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start-epoch', default=0, type=int, dest="start_epoch",
                        help="The epoch to start/resume training at")
    parser.add_argument('--patience', default=1, type=int)


    # Images
    parser.add_argument('--IMG_WIDTH', default=256, type=int,
                        help="The width of the input images to the network")
    parser.add_argument('--IMG_HEIGHT', default=256, type=int,
                        help="The height of the input images to the network")

    # Other arguments
    parser.add_argument("baseline", type=str, choices=["rubi", "san", "updn"],
                        help="Valid baseline net: rubi, san, updn.")

    args = parser.parse_args()

    return args
