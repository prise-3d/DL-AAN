#! /bin/bash

if [ -z "$1" ]
  then
    echo "No argument supplied"
    echo "Need data folder"
    exit 1
fi

if [ -z "$2" ]
  then
    echo "No argument supplied"
    echo "Need output folder"
    exit 1
fi

if [ -z "$3" ]
  then
    echo "No argument supplied"
    echo "Need number of images expected (1 samples images)"
    exit 1
fi

if [ -z "$4" ]
  then
    echo "No argument supplied"
    echo "Need number of samples expected per images"
    exit 1
fi

prefix="p3d_"
build_folder="rawls-tools/build"

data_folder=$1
output_folder=$2
nb_elements=$3
nb_samples=$4

for folder_path in $(ls -d ${data_folder}*/)
do

  IFS='/' read -ra ADDR <<< "${folder_path}"
  folder=${ADDR[-1]}

  # get output expected path
  output_scene_path=$output_folder/$folder
  output_scene_path_fixed=${output_scene_path//\/\//\/}

  if [ ! -d "$output_scene_path_fixed" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    
    mkdir -p $output_scene_path_fixed

    suffix=

    for i in $(seq 1 ${nb_elements}); do

        suffix=$i
        size=${#suffix} 
    
        while [ $size -le 4 ]
        do
            suffix="0${suffix}"
            size=${#suffix} 
        done

        if [ ! -f $output_scene_path_fixed/${folder}_${suffix}.png ]; then
          ./${build_folder}/main/rawls_merge_mean --folder ${folder_path} --samples ${nb_samples} --outfile $output_scene_path_fixed/${folder}_${suffix}.png --random 1
        else
          echo "$output_scene_path_fixed/${folder}_${suffix}.png already generated..."
        fi
    done
  fi
done