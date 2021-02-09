#!/bin/bash

for file in *.mp4
do
fn=${file%.*}
ffmpeg -i "$file" -pix_fmt rgb24 "$fn".gif
done
