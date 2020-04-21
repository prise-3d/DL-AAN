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

  rawls = pypkgs.buildPythonPackage rec {
    pname = "rawls";
    version = "0.1.4";
    src = fetchTarball "https://github.com/prise-3d/rawls/archive/v0.1.4.tar.gz";
    propagatedBuildInputs = with pypkgs; [
      scipy
      pillow
      numpy
    ];
    doCheck = false;
  };

in pkgs.mkShell {

  buildInputs = [
    pkgs.linuxPackages.nvidia_x11

    pypkgs.gym 
    pypkgs.matplotlib
    pypkgs.numpy
    pypkgs.pytorchWithCuda
    pypkgs.torchvision
    pypkgs.scikitimage

    tensorboardX
    rawls
  ];

}

