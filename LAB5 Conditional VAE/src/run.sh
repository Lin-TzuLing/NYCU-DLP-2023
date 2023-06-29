# cyclical train
#python -B main.py \
#	--cuda \
#	--mode train \
#	--batch_size 20 \
#	--beta 0.0 \
#	--kl_anneal_cyclical \
#	--tfr_decay_step 0.01 \
#	--tfr_start_decay_epoch 150 \
#	--exp_name cyclical

# monotonic train
# python main.py \
#	--cuda \
#	--mode train \
#	--batch_size 20 \
#	--beta 0.0 \
#	--tfr_decay_step 0.01 \
#	--tfr_start_decay_epoch 150 \
#	--exp_name monotonic_new

# cyclical test
python -B main.py \
  --cuda \
  --mode test \
  --model_dir './logs/fp/cyclical'\


# monotonic test
python -B main.py \
	--cuda \
	--mode test \
  --model_dir './logs/fp/monotonic'\


# plot result
#RESULT_PATH="./result"
#mkdir -p $RESULT_PATH
#python others/reporter.py
