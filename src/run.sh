python3 Runner.py --batch-size 64 --run-name keras4k_skip --output-path keras_skip --model Unet4k
python3 Runner.py --batch-size 64 --run-name keras4k_noSkip --output-path keras_noSkip --model Unet4k --no-skip
python3 Runner.py --batch-size 64 --run-name gabriele3k_skip --output-path gabriele3k_skip --model Unet3k
python3 Runner.py --batch-size 64 --run-name gabriele3k_noSkip --output-path gabriele3k_noSkip --model Unet3k --no-skip

