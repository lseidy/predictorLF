#python3 Runner.py --batch-size 64 --run-name keras4k_skip_fullEpchs --output-path keras4k_skip_fullEpchs --model Unet4k
#python3 Runner.py --batch-size 64 --run-name keras4k_noSkip_fullEpchs --output-path keras_noSkip_fullEpchs --model Unet4k --no-skip
#python3 Runner.py --batch-size 64 --predictor-size 16 --run-name gabriele3k_noSkip_16x16 --output-path gabriele3k_noSkip_16x16 --model Unet3k --no-skip
#python3 Runner.py --batch-size 64 --predictor-size 16 --run-name gabriele3k_Skip_16x16 --output-path gabriele3k_Skip_16x16 --model Unet3k


python3 Runner.py --batch-size 64 --predictor-size 16 --run-name gabriele3kNoConvTrans_noSkip_16x16 --output-path gabriele3kNoConvTrans_noSkip_16x16 --model Unet3k --no-skip
#python3 Runner.py --batch-size 64 --predictor-size 16 --run-name gabriele3kNoConvTrans_Skip_16x16 --output-path gabriele3kNoConvTrans_Skip_16x16 --model Unet3#k
