from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import re
import codecs
from io import open
from datetime import datetime

import torch


# Method to print lines from a .txt file
def printLines(file, args):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:args.lines]:
        print(line)


# Splits each line(containing the dialogue, speaker,id and unwanted characters) of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)

    # Print sample of the lines in movie_lines.txt
    print("\nSample lines from corpus:")
    printLines(os.path.join(corpus, "movie_lines.txt"), args)

    # Define path to a new file which saves the formatted lines used for training our model
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from the formatted file:")
    printLines(datafile, args)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='ChatBot')

    parser.add_argument('--lines', type=int, default=30,
                        help='number of lines to be printed')

    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')
    args = parser.parse_args()

    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')

    main(args)
