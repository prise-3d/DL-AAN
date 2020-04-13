#include <torch/script.h> // One-stop header.

#include <stdlib.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // please use `autoencoder_convert.py` to convert your model in correct format
    module = torch::jit::load(argv[1]);

    // prepare input data

    // custom input example using vector
    std::vector<float> floatValues;

    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 32; j++)
        for (int k = 0; k < 32; k++)
          floatValues.push_back((float)rand());     
  
    // Create a vector of inputs for load model
    std::vector<torch::jit::IValue> inputs;
    
    auto tensData = torch::tensor(floatValues);
    auto tensDataReshaped = tensData.view({1, 7, 32, 32});

    // pass tensor as input
    inputs.push_back(tensDataReshaped);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();

    std::cout << output.numel() << '\n';

    std::vector<float> xv;
    
    size_t x_size = output.numel();

    // reading output data from buffer example
    auto p = static_cast<float*>(output.data_ptr<float>());
    for(size_t i = 0; i < x_size; i++)
    {
      xv.push_back(p[i]);
    }
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}