jbsub -mem 200G -queue p8_1h -err $1/e.txt -out $1/o.txt -q p8_7d -require k80 python train.py $1/params.json $1
