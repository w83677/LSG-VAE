export CUDA_VISIBLE_DEVICES=5
MODEL=LSG_VAE

DATASET=weather_ltsf
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']
CTX_LEN=96

DATA_DIR=./datasets
LOG_DIR=./log_dir


# if not specify dataset_path, the default path is ./datasets
for PRED_LEN in 96 192 336 720
do 
    python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 1  \
        --data.data_manager.init_args.path ${DATA_DIR} \
        --trainer.default_root_dir ${LOG_DIR} \
        --data.data_manager.init_args.dataset ${DATASET} \
        --data.data_manager.init_args.split_val true \
        --trainer.max_epochs 50 \
        --data.data_manager.init_args.context_length ${CTX_LEN} \
        --data.data_manager.init_args.prediction_length ${PRED_LEN} 
done