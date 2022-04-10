# sample scripts for training vanilla teacher/student models

# CIFAR
python train_teacher.py --model resnet8x4 --trial 0 --gpu_id 0

python train_teacher.py --model resnet32x4 --trial 0 --gpu_id 0

python train_teacher.py --model resnet110 --trial 0 --gpu_id 0

python train_teacher.py --model resnet116 --trial 0 --gpu_id 0

python train_teacher.py --model resnet110x2 --trial 0 --gpu_id 0

python train_teacher.py --model vgg8 --trial 0 --gpu_id 0

python train_teacher.py --model vgg13 --trial 0 --gpu_id 0

python train_teacher.py --model ShuffleV1 --trial 0 --gpu_id 0

python train_teacher.py --model ShuffleV2 --trial 0 --gpu_id 0

python train_teacher.py --model ShuffleV2_1_5 --trial 0 --gpu_id 0

python train_teacher.py --model MobileNetV2 --trial 0 --gpu_id 0

python train_teacher.py --model MobileNetV2_1_0 --trial 0 --gpu_id 0

# WRN-40-1
python train_teacher.py --model resnet38 --trial 0 --gpu_id 0
# WRN-40-2
python train_teacher.py --model resnet38x2 --trial 0 --gpu_id 0
# WRN-16-2
python train_teacher.py --model resnet14x2 --trial 0 --gpu_id 0
# WRN-40-4
python train_teacher.py --model resnet38x4 --trial 0 --gpu_id 0
# WRN-16-4
python train_teacher.py --model resnet14x4 --trial 0 --gpu_id 0

# ImageNet
python train_teacher.py --batch_size 256 --epochs 120 --dataset imagenet --model ResNet18 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23333 --multiprocessing-distributed --dali gpu --trial 0 



