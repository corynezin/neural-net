import math

def inner_product(x,y):
  try:
    inner_prod = sum( x[i]*y[i] for i in range(len(x)) )
  except TypeError:
    inner_prod = x*y
  return(inner_prod)

def activation(a,fcn):
  if fcn == 'relu':
    return max(0,a)  
  if fcn == 'sigmoid':
    return 1.0 / (1.0 + math.exp(-a))

def transpose(A):
  T = []
  M = len(A)
  try:
    N = len(A[0])
  except:
    N = 1
  for n in range(N):
    T.append([0]*M)
  for m in range(M):
    for n in range(N):
      try:
        T[n][m] = A[m][n]
      except:
        T[n][m] = A[m]
  return(T)

def activation_derivative(a,fcn):
  if fcn == 'relu':
    d = 0 if a < 0 else 1
    return(d)
  if fcn == 'sigmoid':
    return activation(a,'sigmoid') * (1.0 - activation(a,'sigmoid'))

def back_prop_learning(example,y,network):
  alpha = 0.1
  input_layer = network[0] 
  hidden_layer = network[1]
  output_layer = network[2]
  a = [[],[],[]]
  h = []
  inn = [[],[]]
  a[0].append(-1.0); a[1].append(-1.0); # Biases
  # Calculate activations
  for x in example:
    a[0].append(x)
  for w in hidden_layer:
    inn[0].append(inner_product(w,a[0]))
    a[1].append(activation(inn[0][-1],'sigmoid'))
  for w in output_layer:
    inn[1].append(inner_product(w,a[1]))
    a[2].append(activation(inn[1][-1],'sigmoid'))
  # Calculate deltas
  delta = [[],[]]
  # Delta at the end ( [g'(aw)][y - g(aw)] )
  for j in range(len(output_layer)):
    delta[1].append(activation_derivative(inn[1][j],'sigmoid') * (y - a[2][j]))

  output_layer_t = transpose(output_layer)
  # Delta one layer down
  for i in range(len(hidden_layer)):
    delta[0].append(activation_derivative(inn[0][i],'sigmoid') * \
      inner_product(output_layer_t[i],delta[1]))

  for i in range(len(output_layer)):
    for j in range(len(output_layer[i])):
      output_layer[i][j] = output_layer[i][j] + (alpha * a[1][j] * delta[1][i])

  for i in range(len(hidden_layer)):
    for j in range(len(hidden_layer[i])):
      hidden_layer[i][j] = hidden_layer[i][j] + (alpha * a[0][j] * delta[0][i])

  network[0] = []
  network[1] = hidden_layer
  network[2] = output_layer
  return(network,(y,a[2][0]))
