jbsub -mem 200G -queue p8_7d -err $1/e.txt -out $1/o.txt -require k80 python train.py $1/params.json $1
