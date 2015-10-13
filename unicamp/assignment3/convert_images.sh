#!/bin/bash

cd ~/Desktop/a3-images

mogrify -format ppm -path ~/Desktop/mlearning/unicamp/assignment3/ppm "*.jpg"
