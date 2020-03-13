let
  rev = "dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b";
  channel = fetchTarball "https://github.com/NixOS/nixpkgs/archive/${rev}.tar.gz";
  config = {
    allowUnfree = true;
    cudaSupport = true;
    enableParallelBuilding = true;
    packageOverrides = pkgs: {
      cudatoolkit = pkgs.cudatoolkit_10;
      cudnn = pkgs.cudnn_cudatoolkit_10;
    };
  };
  pkgs = import channel { inherit config; };

  python = pkgs.python3;
  pypkgs = python.pkgs;
  pytorch = pypkgs.pytorchWithCuda;

  tensorboardX = pypkgs.buildPythonPackage rec {
    pname = "tensorboardX";
    version = "1.8";
    src = fetchTarball "https://github.com/lanpa/tensorboardX/archive/v1.8.tar.gz";
    propagatedBuildInputs = with pypkgs; [
      six
      protobuf
      numpy
    ];
    doCheck = false;
  };

in pkgs.mkShell {

  buildInputs = [
    #pkgs.cudatoolkit
    #pkgs.cudnn
    pkgs.linuxPackages.nvidia_x11

    tensorboardX
    pytorch

    pypkgs.numpy
    pypkgs.torchvision
    pypkgs.matplotlib
    pypkgs.gym 
  ];

  shellHook = ''
      export CFLAGS="-I${pytorch}/${python.sitePackages}/torch/include -I${pytorch}/${python.sitePackages}/torch/include/torch/csrc/api/include"
      export CXXFLAGS=$CFLAGS
      export LDFLAGS="-L${pytorch}/${python.sitePackages}/torch/lib -L$out/${python.sitePackages} -L${pkgs.cudatoolkit}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
      export PYTHONPATH="$PYTHONPATH:build:build/torchRL/mcts:build/torchRL/tube"
      export CUDA_PATH=${pkgs.cudatoolkit}
  '';
      #export OMP_NUM_THREADS=1

}

