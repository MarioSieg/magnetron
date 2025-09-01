#!/usr/bin/env bash

pyinstrument --show-all -r html -o profile.html examples/gpt2/gpt2.py "What is the answer to life?"