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

import mdtraj as md
import numpy as np

class DiversityReward:
    
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
        
    def calc_mdtraj_metrics(self, pdb_path : str):
        try:
            traj = md.load(pdb_path)
            pdb_ss = md.compute_dssp(traj, simplified=True)
            pdb_coil_percent = np.mean(pdb_ss == 'C')
            pdb_helix_percent = np.mean(pdb_ss == 'H')
            pdb_strand_percent = np.mean(pdb_ss == 'E')
            pdb_ss_percent = pdb_helix_percent + pdb_strand_percent 
            pdb_rg = md.compute_rg(traj)[0]
        except IndexError as e:
            print('Error in calc_mdtraj_metrics: {}'.format(e))
            pdb_ss_percent = 0.0
            pdb_coil_percent = 0.0
            pdb_helix_percent = 0.0
            pdb_strand_percent = 0.0
            pdb_rg = 0.0
        return {
            'non_coil_percent': pdb_ss_percent,
            'coil_percent': pdb_coil_percent, # coil
            'helix_percent': pdb_helix_percent, # alpha helix
            'strand_percent': pdb_strand_percent, # beta sheet
            'radius_of_gyration': pdb_rg,
        }
        
    def get_ratio(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path : str,):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file

        """
        
        pdb_path = os.path.join(reference_pdb_path, "sample.pdb")
        calc_ratio = self.calc_mdtraj_metrics(pdb_path)
        
        # save to csv, for debugging
        df = pd.DataFrame(calc_ratio, index=[0])
        df.to_csv(os.path.join(decoy_pdb_dir, "diversity.csv"), index=False)
        
        # We want to 
        # 1. Maximize the ratio of non_coil_percent structure.
        # 2. Maximize the ratio of beta sheet structure.
        # 3. Keep the alpha helix ratio from being "too" high.
        
        return calc_ratio
        
    def get_reward(
            self,
            decoy_pdb_dir: str,
            reference_pdb_path: str,):
        """Run self-consistency on design proteins against reference protein.
        
        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file

        """
        
        # get ratio
        calc_ratio = self.get_ratio(decoy_pdb_dir, reference_pdb_path)
        
        # weighted reward, need to be tuned
        reward = 0
        reward += calc_ratio['non_coil_percent'] * 1.0
        reward += calc_ratio['strand_percent'] * 1.0
        reward -= calc_ratio['helix_percent'] * 0.5
        
        return reward
    
    def get_diversity_rewards(self, pdb_csv_path):
        """Get designability reward from csv file.
        
        pdb_csv_path : str
            Path to the csv file containing the pdb paths
        """
        
        pdb_path_list = pd.read_csv(pdb_csv_path)
        rewards = []
        for pdb_path in pdb_path_list:
            output_dir = os.path.join(pdb_path, "diversity")
            os.makedirs(output_dir, exist_ok=True)
            reward = self.get_reward(output_dir, pdb_path)
            rewards.append(reward)
        
        return rewards

@hydra.main(version_base=None, config_path="../configs", config_name="rl_config")
def run(conf: DictConfig) -> None:

    print('Test diversity')
    start_time = time.time()
    
    # testing path
    pdb_path = "/home/shuaikes/server2/shuaikes/projects/frameflow_rl/inference_outputs/weights/pdb/published/unconditional/run_2024-10-21_01-26-05/length_70/sample_0/"
    output_dir = os.path.join(pdb_path, "diversity")
    os.makedirs(output_dir, exist_ok=True)
    
    # run reward model
    Reward_model = DiversityReward(conf)
    _ = Reward_model.get_reward(output_dir, pdb_path)
    
    # run reward model based on csv file
    # pdb_csv_path = "/home/shuaikes/server2/shuaikes/projects/frameflow_rl/inference_outputs/weights/pdb/published/unconditional/run_2024-10-21_01-26-05/length_70/sample_1/pdb_paths.csv"
    # reward = Reward_model.get_diversity_rewards(pdb_csv_path)
    
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()