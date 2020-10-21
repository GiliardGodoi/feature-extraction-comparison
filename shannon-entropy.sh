#nice python3.6 ShannonEntropy.py -i 1-covid/prep_negative-ncbi.fasta -o processed/1-covid/tmp/shannon-entropy-l0-k12.csv -l 0 -k 12

#nice python3.6 ShannonEntropy.py -i 1-covid/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/1-covid/tmp/shannon-entropy-l1-k12.csv -l 1 -k 12

#nice python3.6 ShannonEntropy.py -i 2-other-viruses/prep_sequences_size_2000_50000.fasta -o processed/2-other-viruses/tmp/shannon-entropy-l0-k12.csv -l 0 -k 12

#nice python3.6 ShannonEntropy.py -i ./3-multi-class/prep_MERS-CoV-ncbi.fasta -o processed/3-multi-class/tmp/shannon-entropy-merscov_l0k12.csv -l 0 -k 12

#nice python3.6 ShannonEntropy.py -i ./3-multi-class/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/3-multi-class/tmp/shannon-entropy-sarscov2_l1k12.csv -l 1 -k 12

#nice python3.6 ShannonEntropy.py -i ./3-multi-class/prep_others.fasta -o processed/3-multi-class/tmp/shannon-entropy-others_l2k12.csv -l 2 -k 12

nice python3.6 ShannonEntropy.py -i 3-multi-class/prep_SARS-CoV-ncbi_virus.fasta -o processed/3-multi-class/tmp/shannon-entropy-sarscov_l3k12.csv -l 3 -k 12

