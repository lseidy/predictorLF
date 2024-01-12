#python3 Runner.py --batch-size 64 --run-name keras4k_skip_fullEpchs --output-path keras4k_skip_fullEpchs --model Unet4k
#python3 Runner.py --batch-size 64 --run-name keras4k_noSkip_fullEpchs --output-path keras_noSkip_fullEpchs --model Unet4k --no-skip
python3 Runner.py --batch-size 64 --run-name gabriele3k_skip_try2 --output-path gabriele3k_skip_try2 --model Unet3k
python3 Runner.py --batch-size 64 --run-name gabriele3k_noSkip_try2 --output-path gabriele3k_noSkip_try2 --model Unet3k --no-skip

