#python3 Runner.py --batch-size 64 --run-name keras4k_skip_fullEpchs --output-path keras4k_skip_fullEpchs --model Unet4k
#python3 Runner.py --batch-size 64 --run-name keras4k_noSkip_fullEpchs --output-path keras_noSkip_fullEpchs --model Unet4k --no-skip
#python3 Runner.py --batch-size 64 --run-name gabriele3k_skip_try2 --output-path gabriele3k_skip_try2 --model Unet3k
#python3 Runner.py --batch-size 64 --run-name gabriele3k_noSkip_try2 --output-path gabriele3k_noSkip_try2 --model Unet3k --no-skip

#python3 Runner.py --batch-size 64 --predictor-size 32 --run-name  MSE_Test_EntropyRGB --output-path MSE_Test_EntropyRGB --model Unet3k


python3 Runner.py --batch-size 64 --predictor-size 8 --run-name satdLong --loss satd --lr 0.00001 --epochs 200


#QUICK DEBUG RUN
#python3 Runner.py --batch-size 64 --predictor-size 32 --limit-train 2 --limit-val 1 --no-wandb