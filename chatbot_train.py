from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import random
from torch.utils.checkpoint import checkpoint
from helpers import batch2TrainData, trimRareWords, loadPrepareData
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from modules import EncoderRNN, LuongAttnDecoderRNN

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


# Average negative log likelihood of the elements that correspond to a 1 in the mask tensor
def maskNLLLoss(inp, target, mask, args):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(args.device)
    return loss, nTotal.item()


# Single training iteration
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
          encoder_optimizer, decoder_optimizer, clip, args):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set args.device options
    input_variable = input_variable.to(args.device)
    lengths = lengths.to(args.device)
    target_variable = target_variable.to(args.device)
    mask = mask.to(args.device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    # noinspection PyArgumentList
    decoder_input = torch.LongTensor([[SOS_token for _ in range(args.batch_size)]])
    decoder_input = decoder_input.to(args.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t],args)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(args.batch_size)]])
            decoder_input = decoder_input.to(args.device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], args)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, print_every, save_every, clip,
               corpus_name, loadFilename, args, writer):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(args.batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, clip, args)
        print_loss += loss
        writer.add_scalar('loss', loss, iteration + 1)

        # Print progress
        if iteration % print_every == 0 or iteration == 1:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, args.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    writer = SummaryWriter('./logs/{0}'.format('chatbot'))

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
    loadFilename = None
    
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, args.hidden_size)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers, dropout)
    # Use appropriate args.device
    encoder = encoder.to(args.device)
    decoder = decoder.to(args.device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    decoder_learning_ratio = 5.0
    n_iteration = args.epochs
    print_every = 100
    save_every = 100

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr * decoder_learning_ratio)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, args.encoder_n_layers, args.decoder_n_layers, save_dir, n_iteration,
               print_every, save_every, clip, corpus_name, loadFilename, args, writer)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Chatbot')

    parser.add_argument('--hidden_size', type=int, default=500,
                        help='size of the feature space')
    parser.add_argument('--attn_model', type=str, default='dot',
                        help='type of attention model: (dot/general/concat)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--min_count',type=int, default=3,
                        help='min_count of a word')
    parser.add_argument('--max_length',type=int, default=10,
                        help='max_length of a sentence')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for Adam optimizer (default: 0.0001)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                        help='teacher_forcing_ratio')
    parser.add_argument('--encoder_n_layers', type=int, default=2,
                        help='layers in the encoder')
    parser.add_argument('--decoder_n_layers', type=int, default=2,
                        help='layers in the decoder')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')

    main(args)
