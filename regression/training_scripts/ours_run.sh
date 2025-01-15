set -e
export EXP_DIR=./results
export N_STEPS=1000
export SERVER_NAME=a4000
export RUN_NAME=run_ours_ce_dev
export N_SPLITS=2
export N_THREADS=4
export DEVICE_ID=0

export CAT_F_PHI=_cat_f_phi

# List of datasets
DATASETS=("uci_boston" "uci_concrete" "uci_energy" "uci_kin8nm" "uci_naval" "uci_power" "uci_wine" "uci_yatch")
# "uci_protein"  # needs N_SPLITS=5

# List of loss functions
LOSSES=("ours.ce")

for LOSS in "${LOSSES[@]}"; do
    export MODEL_VERSION_DIR=our_uci_results/${RUN_NAME}/${LOSS}
    for TASK in "${DATASETS[@]}"; do
        echo "Running task: $TASK with loss: $LOSS"
        
        # Run training
        python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} \
            --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} \
            --config configs/${TASK}.yml # --train_guidance_only

        # Run testing
        python main.py --ni --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} \
            --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} \
            --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test
    done
done

