from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from helpers import indexesFromSentence, normalizeString, trimRareWords, loadPrepareData
from datetime import datetime
from modules import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def evaluate(searcher, voc, sentence, args):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate args.device
    input_batch = input_batch.to(args.device)
    lengths = lengths.to(args.device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, args):
    input_sentence = ''
    while 1:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, args)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    # Load/Assemble voc and pairs
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    save_dir = os.path.join("model", "checkpoints")
    voc, pairs = loadPrepareData(corpus_name, datafile, args.max_length)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, args.min_count)

    # Configure models
    model_name = 'cb_model'
    dropout = 0.1

    # Set checkpoint to load from; set to None if starting from scratch
    checkpoint_iter = args.checkpoint
    loadFilename = os.path.join(save_dir, model_name,
                                '{}-{}_{}'.format(args.encoder_n_layers, args.decoder_n_layers, args.hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.args.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers,
                                  dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate args.device
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    print('Models built and ready to go!')

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, args.device, args.max_length)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, voc, args)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Chatbot')

    parser.add_argument('--hidden_size', type=int, default=500,
                        help='size of the feature space')
    parser.add_argument('--attn_model', type=str, default='dot',
                        help='type of attention model: (dot/general/concat)')
    parser.add_argument('--min_count', type=int, default=3,
                        help='min_count of a word')
    parser.add_argument('--checkpoint', type=int, default=5000,
                        help='checkpoint for the model loading')
    parser.add_argument('--max_length', type=int, default=10,
                        help='max_length of a sentence')
    parser.add_argument('--encoder_n_layers', type=int, default=2,
                        help='layers in the encoder')
    parser.add_argument('--decoder_n_layers', type=int, default=2,
                        help='layers in the decoder')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')

    main(args)
