import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
import esm

class DesignabilityReward:
    
    def __init__(
        self,
        conf: DictConfig,
    ):
        
        self._log = logging.getLogger(__name__)
        self._conf = conf
        self._infer_conf = conf.inference
        # self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples
        
        # Set random seed
        self._rng = np.random.default_rng(self._infer_conf.seed)
        
        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)
        
        # Set-up accelerator
        if torch.cuda.is_available():
            available_gpus = ''.join(
                [str(x) for x in GPUtil.getAvailable(
                    order='memory', limit = 8)])
            self.device = f'cuda:{available_gpus[0]}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')
        
        self._pmpnn_dir = self._infer_conf.pmpnn_dir
        
        # Load ESMFold model
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)
        
    def get_reward(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf
            Writes results in decoy_pdb_dir/sc_results.csv
        """

        # Run PorteinMPNN
        output_path = os.path.join(decoy_pdb_dir, "parsed_pdbs.jsonl")
        process = subprocess.Popen([
            'python',
            f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
            f'--input_path={reference_pdb_path}',
            f'--output_path={output_path}',
        ])
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            decoy_pdb_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            '3',
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        # if self._infer_conf.gpu_id is not None:
        #     pmpnn_args.append('--device')
        #     pmpnn_args.append(str(self._infer_conf.gpu_id))
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            decoy_pdb_dir,
            'seqs',
            'sample.fa'
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'header': [],
            'sequence': [],
        }

        esmf_dir = os.path.join(decoy_pdb_dir, 'esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats('sample.pdb', os.path.join(reference_pdb_path, 'sample.pdb'))
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)

            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(esmf_sample_path)
            mpnn_results['header'].append(header)
            mpnn_results['sequence'].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        print(mpnn_results)
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)
        
        designability_reward = mpnn_results['tm_score'].mean()
        return designability_reward

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output
    
    def get_designability_rewards(self, pdb_csv_path):
        """Get designability reward from csv file.
        
        pdb_csv_path : str
            Path to the csv file containing the pdb paths
        """
        
        pdb_path_list = pd.read_csv(pdb_csv_path)
        rewards = []
        for pdb_path in pdb_path_list:
            sc_output_dir = os.path.join(pdb_path, "self_consistency")
            os.makedirs(sc_output_dir, exist_ok=True)
            reward = self.get_reward(sc_output_dir, pdb_path)
            rewards.append(reward)
        
        return rewards
        


@hydra.main(version_base=None, config_path="../configs", config_name="rl_config")
def run(conf: DictConfig) -> None:

    print('Test designability')
    start_time = time.time()
    
    # Example pdb path
    pdb_path = "/home/shuaikes/server2/shuaikes/projects/frameflow_rl/inference_outputs/weights/pdb/published/unconditional/run_2024-10-21_01-26-05/length_70/sample_1/"
    sc_output_dir = os.path.join(pdb_path, "self_consistency")
    os.makedirs(sc_output_dir, exist_ok=True)
    
    # run reward model
    Reward_model = DesignabilityReward(conf)
    reward = Reward_model.get_reward(sc_output_dir, pdb_path)
    
    # run reward model based on csv file
    # pdb_csv_path = "/home/shuaikes/server2/shuaikes/projects/frameflow_rl/inference_outputs/weights/pdb/published/unconditional/run_2024-10-21_01-26-05/length_70/sample_1/pdb_paths.csv"
    # reward = Reward_model.get_designability_rewards(pdb_csv_path)
     
    
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')
    print(reward)


if __name__ == '__main__':
    run()