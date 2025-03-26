import os
import time
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
from typing import Optional, Union, List
from analysis import utils as au
from analysis import metrics
from data import utils as du
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
import esm
from pathlib import Path
import mdtraj as md
from openfold.np import residue_constants
from tmtools import tm_align
from openfold.utils.superimposition import superimpose
from tqdm import tqdm
import re


class EvalRunner:

    def __init__(
        self,
        conf: DictConfig,
    ):

        self._log = logging.getLogger(__name__)
        self._conf = conf

        # Set random seed
        self._rng = np.random.default_rng(self._conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._conf.pt_hub_dir)

        # Set-up accelerator
        if torch.cuda.is_available():
            available_gpus = "".join(
                [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
            )
            self.device = f"cuda:{available_gpus[0]}"
        else:
            self.device = "cpu"


        self._log.info(f"Using device: {self.device}")

        self._pmpnn_dir = self._conf.pmpnn_dir
        self._foldseek_database = self._conf.foldseek_database

        # Load ESMFold model
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)


    def _calc_bb_rmsd(self, mask, sample_bb_pos, folded_bb_pos):
        aligned_rmsd = superimpose(
            torch.tensor(sample_bb_pos),
            torch.tensor(folded_bb_pos),
            mask.unsqueeze(1).repeat(1, 3).T,
        )
        return aligned_rmsd[1].item()


    def calc_tm_score(self, pos_1, pos_2, seq_1, seq_2):
        tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
        return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2

    def pdbTM(
        self,
        input: Union[str, Path] = None,
        process_id: int = 0,
        save_tmp: bool = False,
        foldseek_path: Optional[Union[Path, str]] = None,
    ) -> Union[float, dict]:
        """
        Calculate pdbTM values with a customized set of parameters by Foldseek.

        Args:
        `input`: Input PDB file or csv file containing PDB paths.
        `process_id`: Used for saving temporary files generated by Foldseek.
        `save_tmp`: If True, save tmp files generated by Foldseek, otherwise deleted after calculation.
        `foldseek_path`: Path of Foldseek binary file for executing the calculations.
                        If you've already installed Foldseek through conda, just use "foldseek"
                        instead of this path.

        CMD args:
        `pdb100`: Path of PDB database created compatible with Foldseek format.
        `output_file`: .m8 file containing Foldseek search results. Deleted if `save_tmp` = False.
        `tmp`: Temporary path when running Foldseek.
        For other CMD parameters and usage, we suggest users go to Foldseek official website
        (https://github.com/steineggerlab/foldseek) or type `foldseek easy-search -h` for detailed information.

        Returns:
        `top_pdbTM`: The highest pdbTM value among the top three targets hit by Foldseek.
        """
        foldseek_database_path = self._foldseek_database
        
        # Handling multiprocessing
        base_tmp_path = "./tmp/"
        tmp_path = os.path.join(base_tmp_path, f"process_{process_id}")
        os.makedirs(tmp_path, exist_ok=True)

        # Check whether input is a directory or a single file
        if ".pdb" in input:
            output_file = f"./{os.path.basename(input)}_{process_id}.m8"

            cmd = f"foldseek easy-search \
                    {input} \
                    {foldseek_database_path} \
                    {output_file} \
                    {tmp_path} \
                    --format-mode 4 \
                    --format-output query,target,evalue,alntmscore,rmsd,prob \
                    --alignment-type 1 \
                    --num-iterations 2 \
                    -e inf \
                    -v 0"

            if foldseek_path is not None:
                cmd.replace("foldseek", {foldseek_path})

            _ = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            result = pd.read_csv(output_file, sep="\t")
            top_pdbTM = round(result["alntmscore"].head(1).max(), 3)

            if save_tmp == False:
                os.remove(output_file)

        else:
            return None

        return top_pdbTM

    def calc_mdtraj_metrics(self, pdb_path: str):
        try:
            traj = md.load(pdb_path)
            pdb_ss = md.compute_dssp(traj, simplified=True)
            pdb_coil_percent = np.mean(pdb_ss == "C")
            pdb_helix_percent = np.mean(pdb_ss == "H")
            pdb_strand_percent = np.mean(pdb_ss == "E")
            pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
            pdb_rg = md.compute_rg(traj)[0]
        except IndexError as e:
            print("Error in calc_mdtraj_metrics: {}".format(e))
            pdb_ss_percent = 0.0
            pdb_coil_percent = 0.0
            pdb_helix_percent = 0.0
            pdb_strand_percent = 0.0
            pdb_rg = 0.0
        return {
            "non_coil_percent": pdb_ss_percent,
            "coil_percent": pdb_coil_percent,  # coil
            "helix_percent": pdb_helix_percent,  # alpha helix
            "strand_percent": pdb_strand_percent,  # beta sheet
            "radius_of_gyration": pdb_rg,
        }

    def run_max_cluster(self, designable_file_path, designable_dir):
        pmpnn_args = [
            "./maxcluster64bit",
            "-l",
            designable_file_path,
            os.path.join(designable_dir, "all_by_all_lite"),
            "-C",
            "2",
            "-in",
            "-Rl",
            os.path.join(designable_dir, "tm_results.txt"),
            "-Tm",
            "0.5",
        ]
        process = subprocess.Popen(
            pmpnn_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, _ = process.communicate()

        # Extract number of clusters
        match = re.search(
            r"INFO\s*:\s*(\d+)\s+Clusters\s+@\s+Threshold\s+(\d+\.\d+)\s+\(\d+\.\d+\)",
            stdout.decode("utf-8"),
        )
        clusters = int(match.group(1))
        cluster_results_path = os.path.join(designable_dir, "cluster_results.txt")
        with open(cluster_results_path, "w") as f:
            f.write(stdout.decode("utf-8"))

        # Extract cluster centroids
        cluster_lines = stdout.decode("utf-8").split("\n")
        centroid_section = False
        for line in cluster_lines:
            if "Centroids" in line:
                centroid_section = True
            if centroid_section:
                match = re.search(r"(?<=\s)(\/[^\s]+\.pdb)", line)
                if match is not None:
                    centroid_path = match.group(1)
                    copy_name = centroid_path.split("/")[-2] + ".pdb"
                    shutil.copy(centroid_path, os.path.join(designable_dir, copy_name))
        return clusters

    def calc_designability(
        self,
        decoy_pdb_dir: str,
        reference_pdb_path: str,
    ):
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
        process = subprocess.Popen(
            [
                "python",
                f"{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
                f"--input_path={reference_pdb_path}",
                f"--output_path={output_path}",
            ]
        )
        _ = process.wait()
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            decoy_pdb_dir,
            "--jsonl_path",
            output_path,
            "--num_seq_per_target",
            "3",
            "--sampling_temp",
            "0.1",
            "--seed",
            "38",
            "--batch_size",
            "1",
        ]

        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                self._log.info(f"Failed ProteinMPNN. Attempt {num_tries}/5")
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(decoy_pdb_dir, "seqs", "sample.fa")

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results = {
            "tm_score": [],
            "bb_rmsd": [],
            "sample_path": [],
            "header": [],
            "sequence": [],
        }

        esmf_dir = os.path.join(decoy_pdb_dir, "esmf")
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = du.parse_pdb_feats(
            "sample.pdb", os.path.join(reference_pdb_path, "sample.pdb")
        )
        for i, (header, string) in enumerate(fasta_seqs.items()):

            # Run ESMFold
            esmf_sample_path = os.path.join(esmf_dir, f"sample_{i}.pdb")
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = du.parse_pdb_feats("folded_sample", esmf_sample_path)
            sample_seq = du.aatype_to_seq(sample_feats["aatype"])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = self.calc_tm_score(
                sample_feats["bb_positions"],
                esmf_feats["bb_positions"],
                sample_seq,
                sample_seq,
            )

            res_mask = torch.ones(sample_feats["bb_positions"].shape[0])

            # print(res_mask.shape)
            # print(sample_feats["bb_positions"].shape)
            # print(esmf_feats["bb_positions"].shape)
            
            bb_rmsd = self._calc_bb_rmsd(
                res_mask, sample_feats["bb_positions"], esmf_feats["bb_positions"]
            )

            mpnn_results["tm_score"].append(tm_score)
            mpnn_results["bb_rmsd"].append(bb_rmsd)
            mpnn_results["sample_path"].append(esmf_sample_path)
            mpnn_results["header"].append(header)
            mpnn_results["sequence"].append(string)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, "sc_results.csv")
        # print(mpnn_results)
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)

        return mpnn_results

    def calc_all_metrics(
        self,
        decoy_pdb_dir: str,
        reference_pdb_path: str,
    ):

        print("Calculating all metrics...")
        print(
            "##########################################################################"
        )
        print("Calculating designability...")
        # calc self-consistency results
        sc_results = self.calc_designability(decoy_pdb_dir, reference_pdb_path)
        print(sc_results)
        print(
            "##########################################################################"
        )
        print("Calculating substructure ratio...")
        # calc substructure ratio
        pdb_path = os.path.join(reference_pdb_path, "sample.pdb")
        calc_ratio = self.calc_mdtraj_metrics(pdb_path)
        # save to csv, for debugging
        df = pd.DataFrame(calc_ratio, index=[0])
        print(df)
        print(
            "##########################################################################"
        )
        print("Calculating novelty (pdbTM)...")
        # calculate novelty (pdbTM)
        value = self.pdbTM(pdb_path, 1)
        print(
            f"TM-Score between {os.path.basename(pdb_path)} and its closest protein in PDB is {value}."
        )
        print(
            "##########################################################################"
        )
        

    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def calc_diversity(self, pdb_csv_path):
        """Get diversity from csv file.

        pdb_csv_path : str
            Path to the csv file containing the pdb paths
        """

        df = pd.read_csv(pdb_csv_path, header=None)

        pdb_path_list = df[0].tolist()

        # log_msg("Running clustering")
        cluster_dir = os.path.join(Path(pdb_csv_path).parent, "cluster")
        os.makedirs(cluster_dir, exist_ok=True)
        # all_metrics = pd.read_csv(metric_path)
        designable_paths = pdb_path_list
        designable_file_path = os.path.join(cluster_dir, "designable_paths.txt")
        with open(designable_file_path, "w") as f:
            f.write("\n".join(designable_paths))

        clusters = self.run_max_cluster(designable_file_path, cluster_dir)

        return clusters


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def run(conf: DictConfig) -> None:

    # Example pdb path
    pdb_path = "/home/shuaikes/Project/protein-evaluation-notebook/example_data/length_70/sample_0/"
    sc_output_dir = os.path.join(pdb_path, "self_consistency")
    os.makedirs(sc_output_dir, exist_ok=True)

    # run reward model
    EvalModel = EvalRunner(conf)
    # EvalModel.calc_all_metrics(sc_output_dir, pdb_path)

    pdb_csv_path = "/home/shuaikes/server2/shuaikes/projects/protein-evaluation-notebook/pdb_path.csv"
    clusters = EvalModel.calc_diversity(pdb_csv_path)


if __name__ == "__main__":
    run()
