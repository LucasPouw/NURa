#!/bin/bash

echo "Clearing/creating the plotting directory"
if [ ! -d "plots" ]; then
  mkdir plots
fi
rm -rf plots/*

echo "Downloading data for Vandermonde matrix..."
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt 
fi

echo "Running script for exercise 1: Poisson distribution ..."
python3 poisson.py > poisson.txt

echo "Running script for exercise 2abc: Vandermonde matrix ..."
python3 vandermonde.py

echo "Running script for timing exercise 2 ..."
python3 timing.py > timing.txt

echo "Generating the pdf"

pdflatex pouw.tex
