import tensorflow as tf
import pandas as pd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import nltk
# this only has to be run once.
nltk.download('punkt')

tf.logging.set_verbosity(tf.logging.INFO)

# Load the vqa CP v2 dataset
print("loading dataset....")
coco_file = pd.read_json("vqacp_v2_train_questions.json")
data = list(coco_file['question'])

UNI_DIR = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/"
BI_DIR = "skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/"
VOCAB_FILE_UNI = UNI_DIR + "vocab.txt"
EMBEDDING_MATRIX_FILE_UNI = UNI_DIR + "embeddings.npy"
CHECKPOINT_PATH_UNI = UNI_DIR + "model.ckpt-501424"
VOCAB_FILE_BI = BI_DIR + "vocab.txt"
EMBEDDING_MATRIX_FILE_BI = BI_DIR + "embeddings.npy"
CHECKPOINT_PATH_BI = BI_DIR + "model.ckpt-500008"

encoder = encoder_manager.EncoderManager()
# load unidirectional model
print("setting up the encoder for the unidirectional model....")
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE_UNI,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE_UNI,
                   checkpoint_path=CHECKPOINT_PATH_UNI)
#load bidirectional model
print("setting up the encoder for the bidirectional model....")
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE_BI,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE_BI,
                   checkpoint_path=CHECKPOINT_PATH_BI)

# Generate Skip-Thought Vectors for each sentence in the data set.
# The encoder will use the concatenation of the unidirectional and bidirectional model
# resulting in a 4800 dimensional encoding of the questions.
print("encoding data...")
encodings = encoder.encode(data, verbose=True)
