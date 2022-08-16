#!/bin/bash

FOLDER='../incoming_to_cut/clear_to_cut/auto_cut'

for f in "$FOLDER"/*; do
    ffmpeg -i "$f" "${f/.avi/.ogv}" && rm "$f"
done
