#!/bin/bash

echo "Clearing/creating the plotting directory"
if [ ! -d "plots" ]; then
  mkdir plots
fi
rm -rf plots/*

echo "Checking if output files already exist"
if [ -e q1a_output.txt ]; then
  echo "Remove q1a_output file"
  rm q1a_output.txt
fi

if [ -e q1bc_output.txt ]; then
  echo "Remove q1bc_output file"
  rm q1bc_output.txt
fi

if [ -e q1d_output.txt ]; then
  echo "Remove q1d_output file"
  rm q1d_output.txt
fi

if [ -e q2a_output.txt ]; then
  echo "Remove q2a_output file"
  rm q2a_output.txt
fi

if [ -e q2b_output.txt ]; then
  echo "Remove q2b_output file"
  rm q2b_output.txt
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

echo "Running script for exercise 1a: ..."
python3 q1a.py > q1a_output.txt

echo "Running script for exercise 1b: ..."
python3 q1b.py > q1b_output.txt

echo "Running script for exercise 1c: ..."
python3 q1c.py > q1c_output.txt

echo "Running script for exercise 1d: ..."
python3 q1d.py > q1d_output.txt

echo "Running script for exercise 2a: ..."
python3 q2a.py > q2a_output.txt

echo "Running script for exercise 2b: ..."
python3 q2b.py > q2b_output.txt

echo "Generating the pdf"

pdflatex pouw.tex
