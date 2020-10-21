
# nice python3.6 concatenate.py ./processed/2-other-viruses/tmp/complex-network-l0k3t10.csv ./processed/1-covid/tmp/complex-network-l1k3t10.csv --out ./processed/2-other-viruses/complex-network_k3t10.csv

# nice python3.6 concatenate.py ./processed/2-other-viruses/tmp/fourier-class-l0-r5.csv ./processed/1-covid/tmp/fourier-class-l1-r5.csv  --out ./processed/2-other-viruses/fourier-class_r5.csv

# nice python3.6 concatenate.py ./processed/2-other-viruses/tmp/shannon-entropy-l0-k12.csv ./processed/1-covid/tmp/shannon-entropy-l1-k12.csv  --out ./processed/2-other-viruses/shannon-entropy_k12.csv

nice python3.6 concatenate_2.py --folder processed/3-multi-class/tmp/ --start complex --out processed/3-multi-class/complex-network_k3t10.csv

nice python3.6 concatenate_2.py --folder processed/3-multi-class/tmp/ --start shannon --out processed/3-multi-class/shannon-entropy_k12.csv

nice python3.6 concatenate_2.py --folder processed/3-multi-class/tmp/ --start fourier --out processed/3-multi-class/fourier-class_r5.csv
