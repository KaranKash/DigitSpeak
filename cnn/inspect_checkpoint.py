from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader("/Users/karan/DigitSpeak/cnn/model/model.ckpt-379")
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape) # Remove this is you want to print only variable names
