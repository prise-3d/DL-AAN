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

prefix="p3d_"
build_folder="rawls-tools/build"

data_folder=$1
output_folder=$2

for folder_path in $(ls -d ${data_folder}*/)
do
  nb_elements=$(ls -l ${folder_path} | grep ${prefix} | wc -l)

  IFS='/' read -ra ADDR <<< "${folder_path}"
  folder=${ADDR[-1]}

  # get output expected path
  output_scene_path=$output_folder/$folder
  output_scene_path_fixed=${output_scene_path//\/\//\/}

  if [ ! -d "$output_scene_path_fixed" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    
    mkdir -p $output_scene_path_fixed

    ./${build_folder}/main/rawls_merge_MON_incr --folder ${folder_path} --output $output_scene_path_fixed --step ${nb_elements} --random 0 --prefix ${folder} --max ${nb_elements} --extension "png"
      
  fi
done