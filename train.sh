CUDA_VISIBLE_DEVICES='1,2' python3 -u train_SEmotion_torch.py

 \
--workers_id 0,1 --batch_size 256 --lr 0.1 \
--stages 100 --data_mode casia --net IR_50 --outdir ./results/IR_50-sface-casia --param_a 0.87 \
--param_b 1.2 2>&1|tee ./logs/IR_50-sface-casia.log



CUDA_VISIBLE_DEVICES='1,2' python3 -u train_softmax.py 

\
 --workers_id 0,1  --lr 0.1 \
--stages 35,65,95 --data_mode casia --net IR_50 --head Softmax --outdir ./results/IR_50-arc-casia \
--target 'lfw' 2>&1|tee ./logs/IR_50-arc-casia.log
