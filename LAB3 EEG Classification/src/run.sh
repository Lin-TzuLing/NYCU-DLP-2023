# create directory
RESULT_PATH="../result/"
SAVE_PATH="../save_model/"

mkdir -p $RESULT_PATH"pic/train_loss"
mkdir -p $RESULT_PATH"pic/train_acc"
mkdir -p $RESULT_PATH"pic/test_acc"
mkdir -p $RESULT_PATH"pic/comparison"
mkdir -p $RESULT_PATH"pic/comparison/acc"
mkdir -p $RESULT_PATH"pic/comparison/loss"
mkdir -p $RESULT_PATH"statistic"
mkdir -p $SAVE_PATH

# EEGNet exp (different lr)
#for lr in 1e-2 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
#do
#    #  different activation function
#    python main.py --model_type eeg --activation_type relu --learning_rate "$lr"
#	  python main.py --model_type eeg --activation_type leaky_relu --learning_rate "$lr"
#	  python main.py --model_type eeg --activation_type elu --learning_rate "$lr"
#done

# DeepConvNet
#for lr in 1e-2 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
#do
#    #  different activation function
#    python main.py --model_type deepconv --activation_type relu --learning_rate "$lr"
#	  python main.py --model_type deepconv --activation_type leaky_relu --learning_rate "$lr"
#	  python main.py --model_type deepconv --activation_type elu --learning_rate "$lr"
#done

# vgg
#for lr in 1e-2 5e-3 2e-3 1e-3 5e-4 2e-4 1e-4
#do
#    #  different activation function
#    python main.py --model_type vgg --activation_type relu --learning_rate "$lr"
#	  python main.py --model_type vgg --activation_type leaky_relu --learning_rate "$lr"
#	  python main.py --model_type vgg --activation_type elu --learning_rate "$lr"
#done

# demo
python main.py --model_type eeg --epochs 500 --activation_type relu --report_every 20 --learning_rate 2e-3 --seed 10
#python main.py --model_type deepconv --activation_type leaky_relu --report_every 20 --learning_rate 1e-2
#python main.py --model_type vgg --activation_type relu --report_every 20 --learning_rate 1e-3 --seed 10