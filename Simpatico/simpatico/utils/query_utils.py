import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.data import Batch
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import pickle
from glob import glob


# results_dir = '/xdisk/twheeler/jgaiser/enamine_HTVS/small_results/'
# pyg_dir = '/xdisk/twheeler/jgaiser/enamine_HTVS/enamine_pygs/'
# active_count_file = '/xdisk/twheeler/jgaiser/enamine_HTVS/mock-results/DUDE_actives_concat.txt'

class ScreenResultsHandler:
    """
    This class handles operations on result files produced by the simpatico.query operation.
    Screening evals (where we have a set of known actives per target) depend on several conventions.
    """
    def __init__(self, results_dir, pyg_dir, active_count_file, active_file_substr):
        self.results_dir = results_dir
        self.pyg_dir = pyg_dir
        self.active_count_file = active_count_file
        
        self.get_total_mol_count()
        self.get_results_data()
        self.get_active_file_index(active_file_substr)
        self.get_start_stop_index()
        self.split_active_decoy_scores()
        
        
    def get_total_mol_count(self):
        mol_total = 0
        
        for f in glob(self.results_dir + '/*.pkl'):
            filename = '.'.join(f.split('/')[-1].split('.')[:-1])
            pyg_filename = pyg_dir + filename + '.pyg'
            g = torch.load(pyg_filename)
            mol_total += (g.ptr.size(0)-1)
            del g
            
        self.mol_total = mol_total
        
    def get_active_file_index(self, substr):
        for i in range(len(self.score_file_list)):
            if substr in self.score_file_list[i]:
                self.active_file_index = i
                break
    
    def get_start_stop_index(self):
        start_stop_index = []
        
        with open(self.active_count_file, 'r') as f:
            for t_i, line in enumerate(f):
                line_content = line.strip().split(',')
                start_stop_vals = [int(x) for x in line_content[1:]]

                if start_stop_vals[-1] == -1:
                    break

                start_stop_index.append(start_stop_vals)
                
        self.start_stop_index = start_stop_index
    
    def get_results_data(self):
        score_file_list = []
        score_file_index = None 
        mol_scores = None 
        mol_index = None

        for f_i, f in enumerate(glob(self.results_dir + '/*.pkl')):
            score_file_list.append(f)
            results = pickle.load(open(f,'rb'))

            if score_file_index is None:
                score_file_index  = [[] for _ in range(len(results))]
                mol_scores  = [[] for _ in range(len(results))]
                mol_index  = [[] for _ in range(len(results))]

            for t_i, (m_score, m_index) in enumerate(results):
                mol_scores[t_i] += m_score
                mol_index[t_i] += m_index
                score_file_index[t_i] += [f_i]*len(m_score)

        for t_i in range(len(score_file_index)):
            t_argsorted_index = torch.tensor(mol_scores[t_i]).argsort(descending=True)

            score_file_index[t_i] = torch.tensor(score_file_index[t_i])[t_argsorted_index]
            mol_scores[t_i] = torch.tensor(mol_scores[t_i])[t_argsorted_index]
            mol_index[t_i] = torch.tensor(mol_index[t_i])[t_argsorted_index]
        
        self.mol_scores = mol_scores
        self.mol_index = mol_index
        self.score_file_index = score_file_index
        self.score_file_list = score_file_list
    
    def split_active_decoy_scores(self):
        split_scores = []

        for t_idx in range(len(self.mol_scores)):
            active_file_mask = self.score_file_index[t_idx] == self.active_file_index
            active_mol_mask = torch.isin(self.mol_index[t_idx], torch.arange(*self.start_stop_index[t_idx]))
            true_active_mask = active_file_mask & active_mol_mask
            active_scores = self.mol_scores[t_idx][true_active_mask]
            decoy_scores = self.mol_scores[t_idx][~true_active_mask]
            split_scores.append([active_scores, decoy_scores])
        
        self.ad_split_scores = split_scores
        
    def get_enrichment_scores(self, sample_ratios=[0.0001, 0.001]):
        enrichment_factors = []
        for t_i, (active_scores, decoy_scores) in enumerate(self.ad_split_scores):
            active_count = self.start_stop_index[t_i][1]-self.start_stop_index[t][0]
            true_ratio = active_count / self.mol_total

            active_mask = torch.hstack((torch.ones(active_scores.size(0)),
                                        torch.zeros(decoy_scores.size(0))))

            all_scores = torch.hstack((active_scores, decoy_scores))
            sorted_mask = active_mask[all_scores.argsort(descending=True)]

            ef_vals = []

            for sr in sample_ratios:
                sample_size = int(mol_count * sr)
                active_count = sorted_mask[:sample_size].sum()
                sample_ratio = active_count / sample_size
                ef = sample_ratio / true_ratio
                ef_vals.append(ef)
            enrichment_factors.append(ef_vals)

        return enrichment_factors 
    
    def get_top_n_smiles(self, target_idx, n=100):
        smiles = [None for _ in range(n)]

        top_mol_scores = self.mol_scores[target_idx][:n]
        top_mol_index = self.mol_index[target_idx][:n]
        top_score_file_index = self.score_file_index[target_idx][:n]

        for f_i in top_score_file_index.unique():
            result_file = self.score_file_list[f_i]
            filename = '.'.join(result_file.split('/')[-1].split('.')[:-1])
            smile_file = self.pyg_dir + filename + '_out.ism'

            mol_ranks_from_file = torch.where(top_score_file_index == f_i)[0]
            smile_index = top_mol_index[mol_ranks_from_file]

            with open(smile_file, 'r') as smile_in:
                for l_i, line in enumerate(smile_in):
                    if l_i in smile_index:
                        smile_rank = mol_ranks_from_file[smile_index == l_i].item()
                        smiles[smile_rank] = line.rstrip()
        return smiles
    
    def draw_smiles(self, smiles_list, n=8):
        molecules = [Chem.MolFromSmiles(x) for x in smiles_list]
        img = Draw.MolsToGridImage(molecules[:n],molsPerRow=2,subImgSize=(400,400)) 
        return img
    

        
# screen_results = ScreenResultsHandler(results_dir, pyg_dir, active_count_file, 'DUDE_')

