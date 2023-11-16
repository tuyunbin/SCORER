# Self-Supervised Cross-View Representation Reconstruction for Change Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Self-Supervised Cross-View Representation Reconstruction for Change Captioning"](https://openaccess.thecvf.com/content/ICCV2023/papers/Tu_Self-supervised_Cross-view_Representation_Reconstruction_for_Change_Captioning_ICCV_2023_paper.pdf), which has appeared as a regular paper in ICCV 2023. 

## We illustrate the training details as follows:

## Installation
1. Clone this repository
2. cd SCORER
1. Make virtual environment with Python 3.8 
2. Install requirements (`pip install -r requirements.txt`)
3. Setup COCO caption eval tools ([github](https://github.com/mtanti/coco-caption)) 
4. An NVIDA 3090 GPU or others.

## Data
1. Download data from here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing)
```
python google_drive.py 1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe clevr_change.tar.gz
tar -xzvf clevr_change.tar.gz
```
Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

We are providing the preprocessed data here: [google drive link](https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing).
You can skip the procedures explained below and just download them using the following command:
```
python google_drive.py 1FA9mYGIoQ_DvprP6rtdEve921UXewSGF ./data/clevr_change_features.tar.gz
cd data
tar -xzvf clevr_change_features.tar.gz
```

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files using caption annotations:
```
python scripts/preprocess_captions_transformer.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --split_json ./data/splits.json --output_vocab_json ./data/vocab.json --output_h5 ./data/labels.h5
```

## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train.py --cfg configs/dynamic/transformer.yaml
```

## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test.py --cfg configs/dynamic/transformer.yaml  --snapshot 10000 --gpu 1
```
The command above will take the model snapshot at 10000th iteration and run inference using GPU ID 1.

## Evaluation
* Caption evaluation

Run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/SCORER+CBR/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/SCORER+CBR/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/SCORER+CBR/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@inproceedings{tu2023self,
  title={Self-supervised Cross-view Representation Reconstruction for Change Captioning},
  author={Tu, Yunbin and Li, Liang and Su, Li and Zha, Zheng-Jun and Yan, Chenggang and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2805--2815},
  year={2023}
}
```

## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!


