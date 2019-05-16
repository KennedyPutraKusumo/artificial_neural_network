from ann import ArtificialNeuralNetwork as ann
import numpy as np

ann_1 = ann([900, 12, 10])
sensory_input = np.zeros(900)
output = ann_1.predict(sensory_input)
print(output)
