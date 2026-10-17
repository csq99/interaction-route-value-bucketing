# DeiT Shared Interaction Adapter: Epoch 10 to 15

Independent continuation of the existing 40k/5k shared interaction run.
The first script resumes epoch 10, trains through epoch 15, saves epochs 12 and 15, retrains the Router from epoch-15 Adapter, and runs the 5k dynamic evaluation plus GPU-forward benchmark. The second script evaluates all legal fixed routes for epochs 12 and 15 on 512 rows.

This remains the 40k train / 5k test code-split protocol, not a complete ImageNet training reproduction. It does not overwrite previous directories.

Upload this whole directory to:

/HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-40k-e10-to-e15-run

Do not upload model, parquet data, or checkpoint files; the resume checkpoint is already on the server.

On the server:

cd /HOME/pxyai/pxyaih_0002/CSQ/ASTER/ASTER-main/aster-deit-interaction-multibudget-40k-e10-to-e15-run
sed -i 's/\\r$//' run_e10_to_e15.sh run_fixed_quality_e12_e15.sh
chmod +x run_e10_to_e15.sh run_fixed_quality_e12_e15.sh
sbatch run_e10_to_e15.sh

After that job completes, submit: sbatch run_fixed_quality_e12_e15.sh
