# Diffusion Autoencoders
## Setup 
create conda environment with python 3.11

conda activate environment 

pip install -r requirements.txt

## Training Diff-AE

accelerate launch --multi_gpu train.py --config=./configs/yourconfig.yaml --bs=batch_size --stage=stage_to_train --three_d 

--three_d only when using 3d date

## Evaluating Diff-AE

