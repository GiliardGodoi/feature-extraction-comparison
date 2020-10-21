#nice python3.6 FourierClass.py -i 1-covid/prep_negative-ncbi.fasta -o processed/1-covid/tmp/fourier-class-l0-r5.csv -l 0 -r 5

#nice python3.6 FourierClass.py -i 1-covid/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/1-covid/tmp/fourier-class-l1-r5.csv -l 1 -r 5

#nice python3.6 FourierClass.py -i 2-other-viruses/prep_sequences_size_2000_50000.fasta -o processed/2-other-viruses/tmp/fourier-class-l0-r5.csv -l 0 -r 5

nice python3.6 FourierClass.py -i ./3-multi-class/prep_MERS-CoV-ncbi.fasta -o processed/3-multi-class/tmp/fourier-class-merscov_l0r5.csv -l 0 -r 5

nice python3.6 FourierClass.py -i ./3-multi-class/prep_SARS-CoV-2-ncbi_virus.fasta -o processed/3-multi-class/tmp/fourier-class-sarscov2_l1r5.csv -l 1 -r 5

nice python3.6 FourierClass.py -i ./3-multi-class/prep_others.fasta -o processed/3-multi-class/tmp/fourier-class-others_l2r5.csv -l 2 -r 5

nice python3.6 FourierClass.py -i ./3-multi-class/prep_SARS-CoV-ncbi_virus.fasta -o processed/3-multi-class/tmp/fourier-class-sarscov_l3r5.csv -l 3 -r 5
