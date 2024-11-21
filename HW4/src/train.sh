#! /bin/bash

python train_disable_delayed_update.py > /dev/null 2>&1 &
python train_disable_smoothing.py > /dev/null 2>&1 &
python train_single_q.py > /dev/null 2>&1 &

num_training_main=3
for ((i=0;i<$num_training_main;i++))
do
    python main.py > /dev/null 2>&1 &
done

wait
echo "All training processes are done!"
