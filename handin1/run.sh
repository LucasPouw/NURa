#!/bin/bash

echo "Clearing/creating the plotting directory"
if [ ! -d "plots" ]; then
  mkdir plots
fi
rm -rf plots/*

echo "Checking if output files already exist"
if [ -e poisson_output.txt ]; then
  echo "Remove poisson_output file"
  rm poisson_output.txt
fi

if [ -e vandermonde_output.txt ]; then
  echo "Remove vandermonde_output file"
  rm vandermonde_output.txt
fi

if [ -e timing_output.txt ]; then
  echo "Remove timing_output file"
  rm timing_output.txt
fi

echo "Checking if .pdf, .log, .out and .aux already exist"
if [ -e pouw.aux ]; then
  echo "Remove pouw.aux file"
  rm pouw.aux
fi

if [ -e pouw.log ]; then
  echo "Remove pouw.log file"
  rm pouw.log
fi

if [ -e pouw.out ]; then
  echo "Remove pouw.out file"
  rm pouw.out
fi

if [ -e pouw.pdf ]; then
  echo "Remove pouw.pdf file"
  rm pouw.pdf
fi

echo "Downloading data for Vandermonde matrix..."
if [ ! -e Vandermonde.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/Vandermonde.txt 
fi

echo "Running script for exercise 1: Poisson distribution ..."
python3 poisson.py > poisson_output.txt

echo "Running script for exercise 2abc: Vandermonde matrix ..."
python3 vandermonde.py > vandermonde_output.txt

echo "Running script for timing exercise 2, this should take ~20 sec ..."
python3 timing.py > timing_output.txt

echo "Generating the pdf"

pdflatex pouw.tex
