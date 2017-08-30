import helper_nn
import fileio

network = fileio.read_neural_net('sample.NNWDBC.init')
examples,classes = fileio.read_examples('wdbc.train')
n = 0
for i in range(100):
  for example,label in zip(examples,classes):
    network,err = helper_nn.back_prop_learning(example,label,network)

#example,label = fileio.read_examples('wdbc.mini_train')
#for i in range(100000):
#network,err = helper_nn.back_prop_learning(examples[0],classes[0],network)

if False:
  with open('output.txt','w') as text_file:
    text_file.write('30 5 1\n')
    for layer in network:
      for node in layer:
        for (n,elem) in zip(range(len(node)),node):
          text_file.write('{:.3f}'.format(elem))
          if n < len(node) - 1:
            text_file.write(' ')
        text_file.write('\n')

for layer in network:
  for node in layer:
    for elem in node:
      print('{:.3f}'.format(elem),end=' ')
    print('\n')
