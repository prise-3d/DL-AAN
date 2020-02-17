# Autoencoder Adversarial Network

## Requirements

```bash
git clone --recursive https://github.com/prise-3d/DL-AAN.git
```

```bash
pip install -r requirements.txt
```

Generate necessary noises data from `.rawls` files
```
bash run/generate_data.sh `scenes_folder` `noises_output_folder` `nb_images` `nb_samples_per_image`
```

Generate references data from `.rawls` files
```
bash run/generate_reference.sh `scenes_folder` `reference_output_folder`
```

## How to use ?

### Generate train and test data from scenes data

This python script generates `nb` tiles chosen randomly for each noisy images of `noises_folder` for training and testing dataset.

```
data_processing/generate_dataset.py --noises `noises_folder` --references `references_folder` --nb `number_of_images` --output `output_folder`
```

### Run and train model

```
python train_aan.py --folder `dataset_folder` --batch_size 128 --save aan1 
```

*Note:* model will be saved into `saved_models` folder.

## Relaunch and train model

```
python train_aan.py --folder `dataset_folder` --batch_size 128 --save aan1 --load aan1 
```

## License

[MIT](LICENSE)