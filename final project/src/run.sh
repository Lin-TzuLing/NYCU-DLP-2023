# train meta model with classfication and keypoint coord prediction, no pretrained weight
# python main_lin.py --mode train 
# python main_lin.py --mode train --img_size 384 288

# train baseline model with only classfication task trained independently, no pretrained weight
# python main_lin.py --mode baseline_train
# python main_lin.py --mode baseline_train --img_size 384 288

# train meta model with pretrained weight
# python main_lin.py --mode train --pretrained 
# python main_lin.py --mode baseline_train --pretrained 
# python main_lin.py --mode train --pretrained --img_size 384 288 --joint_epochs 200
# python main_lin.py --mode baseline_train --pretrained --img_size 384 288 



# test without pretrained weight
python main_lin.py --mode test --task classification
python main_lin.py --mode test --task keypoint 
# python main_lin.py --mode baseline_test --task classification
# python main_lin.py --mode test --task classification --img_size 384 288
# python main_lin.py --mode test --task keypoint --img_size 384 288
# python main_lin.py --mode baseline_test --task classification --img_size 384 288

# test with pretrained weight
# python main_lin.py --mode test --task classification --pretrained 
# python main_lin.py --mode test --task keypoint --pretrained
# python main_lin.py --mode baseline_test --task classification --pretrained
# python main_lin.py --mode test --task classification --pretrained --img_size 384 288
# python main_lin.py --mode test --task keypoint --pretrained --img_size 384 288
# python main_lin.py --mode baseline_test --task classification --pretrained --img_size 384 288




# python main_lin.py --mode train --arch resnet50 --joint_epochs 300 
# python main_lin.py --mode test --arch resnet50 
# python main_lin.py --mode baseline --arch resnet50 