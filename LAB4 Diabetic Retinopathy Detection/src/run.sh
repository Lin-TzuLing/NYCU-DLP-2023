
# create directory
RESULT_PATH="../result/"
SAVE_PATH="../model_weight/"
mkdir -p $RESULT_PATH"pic"
mkdir -p $RESULT_PATH"statistic"
mkdir -p $SAVE_PATH


#python main.py --epochs 20 --model_type resnet18 --batch_size 12 --pretrain_flag  --seed 123
#python main.py --epochs 20 --model_type resnet18 --batch_size 12 --seed 123
#python main.py --epochs 20 --model_type resnet50  --batch_size 6 --pretrain_flag --seed 123
#python main.py --epochs 20 --model_type resnet50  --batch_size 6 --seed 123

python main.py --demo --model_type resnet18 --batch_size 12 --pretrain_flag --seed 123