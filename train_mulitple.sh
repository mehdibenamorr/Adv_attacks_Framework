#!/bin/bash

configs=('tests/FFN_1Layer_100' 'tests/FFN_1Layer_150' 'tests/FFN_1Layer_200' 'tests/FFN_1Layer_250' 'tests/FFN_1Layer_300' 'tests/FFN_1Layer_350' 'tests/FFN_1Layer_400')

for configfile in "${configs[@]}"; do
	echo "$configfile"
	python train_model.py --config-file "$configfile"
done
