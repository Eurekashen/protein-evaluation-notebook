foldseek easy-search \
         /home/shuaikes/Project/protein-evaluation-notebook/example_data/length_70/sample_0/sample.pdb \
         /home/shuaikes/Project/protein-evaluation-notebook/foldseek_database/pdb \
         sample.pdb.m8 \
         /home/shuaikes/Project/tmp/process_1 \
         --format-mode 4 \
         --format-output query,target,evalue,alntmscore,rmsd,prob \
         --alignment-type 1 \
         --num-iterations 2 \
         -e inf \
         -v 0