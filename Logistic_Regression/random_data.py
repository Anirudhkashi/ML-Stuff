import random

with open("./data/data.txt", "w") as f:
	for i in range(1000):
		f.write(str(float(random.random())) + "," + str(float(random.random())) + "," + str(float(random.random())) + "," + str(float(random.randint(0, 1))) + "\n")