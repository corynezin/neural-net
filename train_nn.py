import helper_nn
import fileio

filename = input('Enter file containing network to be trained.\n')
network = fileio.read_neural_net(filename)
filename = input('Enter file containing training examples.\n')
examples,classes = fileio.read_examples(filename)
filename = input('Enter the output file for the network.\n')

for i in range(100):
  for example,label in zip(examples,classes):
    network,err = helper_nn.back_prop_learning(example,label,network)

with open(filename,'w') as text_file:
  text_file.write('30 5 1\n')
  for layer in network:
    for node in layer:
      for (n,elem) in zip(range(len(node)),node):
        text_file.write('{:.3f}'.format(elem))
        if n < len(node) - 1:
          text_file.write(' ')
      text_file.write('\n')

print('Neural net has finished training.')
