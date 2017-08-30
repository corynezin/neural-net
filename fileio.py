def read_neural_net(filename):
  with open(filename,'r') as nfile:
    line_num = 0
    for line in nfile:
      if line_num == 0:
        input_layers,hidden_layers,output_layers = line.split()
        input_layers = int(input_layers)
        hidden_layers = int(hidden_layers)
        output_layers = int(output_layers)
        network = [[],[],[]]
      elif line_num <= hidden_layers:
        str_list = line.split()
        flt_list = []
        for num in str_list:
          flt_list.append(float(num))
        network[1].append( flt_list )
      else:
        str_list = line.split()
        flt_list = []
        for num in str_list:
          flt_list.append(float(num))
        network[2].append( flt_list )
      line_num = line_num + 1
  return network

def read_examples(filename):
  with open(filename,'r') as efile:
    line_num = 0
    example_list = []
    class_list = []
    for line in efile:
      if line_num == 0:
        examples,features,outputs = line.split()
        examples = int(examples)
        features = int(features)
        outputs = int(outputs)
      elif line_num <= examples:
        str_list = line.split()
        flt_list = []
        for num in str_list[0:-1]:
          flt_list.append(float(num))
        example_list.append(flt_list)
        class_list.append(int(str_list[-1]))
      line_num = line_num + 1
  return example_list,class_list
