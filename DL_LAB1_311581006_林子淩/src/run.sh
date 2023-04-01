RESULT_PATH='../result/'
mkdir -p $RESULT_PATH

# different hidden_dim
#EXP_TITLE='exp_HiddenDim'
#EXP_PATH="$RESULT_PATH$EXP_TITLE"
#python main.py -exp_path $EXP_PATH -hidden_dim 10 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 20 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 30 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 40 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 60 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 70 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 80 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 90 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 100 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#
#python main.py -exp_path $EXP_PATH -hidden_dim 10 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 20 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 30 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 40 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 60 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 70 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 80 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 90 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 100 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10


# different lr
#EXP_TITLE='exp_Lr'
#EXP_PATH="$RESULT_PATH$EXP_TITLE"
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type sigmoid -lr 5e-2 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type sigmoid -lr 1e-2 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type sigmoid -lr 5e-3 -exp_iterations 10
#
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type sigmoid -lr 1e-1 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type sigmoid -lr 5e-2 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type sigmoid -lr 1e-2 -exp_iterations 10
#python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type sigmoid -lr 5e-3 -exp_iterations 10

# different activation function
EXP_TITLE='exp_Activation'
EXP_PATH="$RESULT_PATH$EXP_TITLE"
python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type none -lr 1e-1 -exp_iterations 10
python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type linear -activation_type relu -lr 1e-1 -exp_iterations 10
python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type none -lr 1e-1 -exp_iterations 10
python main.py -exp_path $EXP_PATH -hidden_dim 50 -data_type xor -activation_type relu -lr 1e-1 -exp_iterations 10








