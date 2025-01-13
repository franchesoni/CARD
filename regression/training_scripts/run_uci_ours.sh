export EXP_DIR=./results
export N_STEPS=1000
export SERVER_NAME=a4000
export RUN_NAME=run_ours
export TASK=uci_concrete
export N_SPLITS=2
export N_THREADS=4
export DEVICE_ID=0

export CAT_F_PHI=_cat_f_phi

export LOSS=ours.ce
export MODEL_VERSION_DIR=our_uci_results/${RUN_NAME}/${LOSS}


python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test
# export LOSS=ours.mdn
# python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
# python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test
# export LOSS=ours.pinball
# python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
# python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test