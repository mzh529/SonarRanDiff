# SonarRanDiff: Enhancing Sonar Image Segmentation through Random Fusion on Diffusion
## Environment
Use the following command to obtain the conda environment:
```bash
pip install -r requirement.txt
```
## Datasets
1. ISIC:Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "data" should be like:
~~~
data
|   ----ISIC
|       ----Test
|       |   |   ISBI2016_ISIC_Part1_Test_GroundTruth.csv
|       |   |   
|       |   ----ISBI2016_ISIC_Part1_Test_Data
|       |   |       ISIC_0000003.jpg
|       |   |       .....
|       |   |
|       |   ----ISBI2016_ISIC_Part1_Test_GroundTruth
|       |           ISIC_0000003_Segmentation.png
|       |   |       .....
|       |           
|       ----Train
|           |   ISBI2016_ISIC_Part1_Training_GroundTruth.csv
|           |   
|           ----ISBI2016_ISIC_Part1_Training_Data
|           |       ISIC_0000000.jpg
|           |       .....
|           |       
|           ----ISBI2016_ISIC_Part1_Training_GroundTruth
|           |       ISIC_0000000_Segmentation.png
|           |       .....
~~~
2. Sonar dataset：You can obtain it by emailing 230238565@seu.edu.cn.

## Training
run: ``python scripts/segmentation_train.py --data_name ISIC --data_dir *input data direction* --out_dir *output data direction* --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``
## Sampling
run: ``python scripts/segmentation_sample.py --data_name ISIC --data_dir *input data direction* --out_dir *output data direction* --model_path *saved model* --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``
## Evaluation
run ``python scripts/segmentation_env.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*``
