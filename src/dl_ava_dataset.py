import os

with open('../data/ava_speech_file_names_v1.txt', "r") as fd:
    for line in fd:
        url = "https://s3.amazonaws.com/ava-dataset/trainval/" + line
        command = "start \"\" " + url
        os.system(command)