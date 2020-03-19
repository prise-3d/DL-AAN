with import <nixpkgs> {};
#with import (fetchTarball "https://github.com/juliendehos/nixpkgs/archive/6afeb1dd1ff5c79b82d0e0ef697fe9cffc434706.tar.gz") {};

let 

  tensorboardX = pkgs.python3Packages.buildPythonPackage rec {
    pname = "tensorboardX";
    version = "1.8";
    src = fetchTarball "https://github.com/lanpa/tensorboardX/archive/v1.8.tar.gz";
    propagatedBuildInputs = with pkgs.python3Packages; [
      six
      protobuf
      numpy
    ];
    doCheck = false;
  };

  rawls = pypkgs.buildPythonPackage rec {
    pname = "rawls";
    version = "0.1.2";
    src = fetchTarball "https://github.com/prise-3d/rawls/archive/v0.1.2.tar.gz";
    propagatedBuildInputs = with pypkgs; [
      scipy
      pillow
      numpy
    ];
    doCheck = false;
  };

in

(python3.withPackages ( ps: with ps; [
  numpy
  torch
  torchvision
  matplotlib
  tensorflow
  tensorboardX
  rawls
  gym 
])).env

