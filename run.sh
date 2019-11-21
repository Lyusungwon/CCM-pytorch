TFORCE=0.8
RUNDIR='tf0.8'

export NCCL_LL_THRESHOLD=0
python -m torch.distributed.launch --nproc_per_node=4 trainer.py \
--teacher_forcing ${TFORCE} \
--exp_name ${RUNDIR}