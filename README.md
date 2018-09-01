# Cycle-GAN-with-Unet-as-GENERATOR

Install TensorboardX 
Python3 and Pytorch

Go to the following link:

http://people.ee.ethz.ch/~ihnatova/#dataset

Click on "Download patches for CNN training (6.2 GB)"

Unzip the file and use the following path as data root argument: ./dped/iphone/

Run train.py with the following:

python main.py --dataroot path/to/data/dped/iphone/ --log_dir path_to_log_dir
