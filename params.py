import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str, help='Mode (train | test)')
    parser.add_argument('--model', default='baseline', type=str, help='Model that will be used')
    parser.add_argument('--answer_type', default='number', type=str, help='answer type (number | yes-no | other)')

    # Directories
    parser.add_argument('--dir_images', default='Images/')
    parser.add_argument('--dir_data', default='Data')
    parser.add_argument('--dir_model', default='Models/', help='Path to project data')

    # Images
    parser.add_argument('--IMG_WIDTH', default=256)
    parser.add_argument('--IMG_HEIGHT', default=256)

    # Training
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--freeVision', default=False, type=bool)
    parser.add_argument('--freeComment', default=True, type=bool)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--nepochs', default=50, type=int)

    # Model params
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--patience', default=1, type=int)

    # Test
    parser.add_argument('--model_path', default='Models/best-retrieval-kgm-author.pth.tar', type=str)
    parser.add_argument('--path_results', default='Results/', type=str)
    parser.add_argument('--no_cuda', action='store_true')

    return parser