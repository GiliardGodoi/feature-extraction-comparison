#nice python3.6 ComplexNetworksClass.py -i 1-covid/prep_negative-ncbi.fasta -o processed/1-covid/tmp/complex-network-l0k3t10.csv -l 0 -k 3 -t 10

#nice python3.6 ComplexNetworksClass.py -i 1-covid/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/1-covid/tmp/complex-network-l1k3t10.csv -l 1 -k 3 -t 10

#nice python3.6 ComplexNetworksClass.py -i 2-other-viruses/prep_sequences_size_2000_50000.fasta  -o processed/2-other-viruses/tmp/complex-network-l0k3t10.csv -l 0 -k 3 -t 10


nice python3.6 ComplexNetworksClass.py -i ./3-multi-class/prep_MERS-CoV-ncbi.fasta -o processed/3-multi-class/tmp/complex-network-merscov_l0kk3t10.csv -l 0 -k 3 -t 10

nice python3.6 ComplexNetworksClass.py -i ./3-multi-class/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/3-multi-class/tmp/complex-network-sarscov2_l1k3t10.csv -l 1 -k 3 -t 10

nice python3.6 ComplexNetworksClass.py -i ./3-multi-class/prep_others.fasta -o processed/3-multi-class/tmp/complex-network-others_l2k3t10.csv -l 2 -k 3 -t 10

nice python3.6 ComplexNetworksClass.py -i ./3-multi-class/prep_SARS-CoV-ncbi_virus.fasta -o processed/3-multi-class/tmp/complex-network-sarscov_l3k3t10.csv -l 3 -k 3 -t 10

