jbsub -cores 1+1 -err $1/e.txt -out $1/o.txt -q p8_12h python train.py $1/params.json
