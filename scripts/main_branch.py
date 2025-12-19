import yaml
import sys 
import os
sys.path.append(os.path.abspath("../src/"))
from twostep_deeponet_trunk import *
from twostep_deeponet_branch import *
import flax
import matplotlib.pyplot as plt

if __name__ == '__main__':

	print(jax.devices())

	stream = open("input.yaml", 'r')
	dictionary = yaml.safe_load(stream)
	
    # Train the branch network
	print("Training branch network...")
	train_branchnet(dictionary)