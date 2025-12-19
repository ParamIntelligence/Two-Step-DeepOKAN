import yaml
import sys 
import os
sys.path.append(os.path.abspath("../src/"))
from twostep_deeponet_trunk import *
import flax
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print(jax.devices())

	stream = open("input.yaml", 'r')
	dictionary = yaml.safe_load(stream)

    # Train the trunk network
	print("Training trunk network...")
	train_trunknet(dictionary)