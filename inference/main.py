import yaml
import sys 
import os
sys.path.append(os.path.abspath("../saved_models/"))
from predict import *
import flax
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print(jax.devices())

	stream = open("input.yaml", 'r')
	dictionary = yaml.safe_load(stream)

	# Predict the solutions
	infer_solution(dictionary)