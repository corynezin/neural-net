def inner_product(x,y):
  inner_prod = sum( x[i]*y[i] for i in range(len(w)) )

def activation(a,fcn):
  if fcn == 'relu':
    return max(0,a)
  

def back_prop_learning(examples,network):
  input_layer = network[0]
  for elem in input_layer:
    hidden_output = []
    for w in network[1]: # Hidden Layer
      x = input_layer[0]
      b = input_layer[1]
      y = inner_product(w,x) + b
      a = activation(a,'relu')
      hidden_output.append(a)
    for w in network[2] # Output Layer
      x = inner_product(w,hidden_output)
      
