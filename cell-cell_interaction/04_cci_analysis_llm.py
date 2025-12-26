import os
import json
import logging
import random
import time
import re
from datetime import datetime
from functools import lru_cache

import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import torch
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from crewai import Agent, LLM, Task, Crew, Process

import itertools
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.environ["GOOGLE_CLOUD_PROJECT"] = "ringed-codex-468710-j7"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# -----------------------------------------------------------------------------
# Load spatial data
# -----------------------------------------------------------------------------
logger.info("Loading IPF ST dataset...")
file_path = '/../my_new_data.h5ad'
adata = sc.read_h5ad(file_path)
print(adata.obs["cell.type"].value_counts())

if "cell.type" not in adata.obs:
    raise ValueError("Dataset must have 'cell.type' column.")

# For consistency in code, create a 'Cell_class' column copying 'cell.type' (if needed)
adata.obs['Cell_class'] = adata.obs['cell.type']

# -----------------------------------------------------------------------------
# Ligand–Receptor database
# -----------------------------------------------------------------------------
logger.info("Loading ligand–receptor reference...")
lr_pairs = pd.read_csv("/../combined_ligand_receptor_pairs.csv")  # supply your own ligand-receptor file

genes_in_data = set(adata.var_names)
lr_pairs = lr_pairs[
    lr_pairs["ligand_gene"].isin(genes_in_data) &
    lr_pairs["receptor_gene"].isin(genes_in_data)
].reset_index(drop=True)
logger.info(f"Filtered to {len(lr_pairs)} ligand–receptor pairs in dataset.")

# Build a dictionary ligand -> list of receptors (for faster lookup)
ligand_receptor_dict = {}
for _, row in lr_pairs.iterrows():
    ligand = row['ligand_gene']
    receptor = row['receptor_gene']
    if ligand not in ligand_receptor_dict:
        ligand_receptor_dict[ligand] = []
    ligand_receptor_dict[ligand].append(receptor)

# -----------------------------------------------------------------------------
# Spatial graph & GAE embedding
# -----------------------------------------------------------------------------
def build_spatial_knn_graph(adata, k=10):
    coords = adata.obs[["original_x", "original_y"]].values
    tree = cKDTree(coords)
    rows, cols = [], []
    for i, pt in enumerate(coords):
        _, idxs = tree.query(pt, k=k+1)
        for nb in idxs[1:]:
            rows.append(i)
            cols.append(nb)
    mat = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    return mat.maximum(mat.transpose())

logger.info("Building spatial KNN graph...")
knn_adjacency = build_spatial_knn_graph(adata, k=10)

logger.info("Running PCA...")
n_pcs = 30
sc.pp.pca(adata, n_comps=n_pcs, random_state=RANDOM_SEED)
X_pca = adata.obsm["X_pca"]

logger.info("Preparing data for GAE...")
edge_index, _ = from_scipy_sparse_matrix(knn_adjacency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_pyg = Data(x=torch.tensor(X_pca, dtype=torch.float32), edge_index=edge_index).to(device)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = GAE(Encoder(in_channels=n_pcs, out_channels=20)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

logger.info("Training GAE...")
model.train()
for epoch in range(30):
    optimizer.zero_grad()
    z = model.encode(data_pyg.x, data_pyg.edge_index)
    loss = model.recon_loss(z, data_pyg.edge_index)
    loss.backward()
    optimizer.step()
    logger.info(f"Epoch {epoch+1}/30, Loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    spatial_embeddings = model.encode(data_pyg.x, data_pyg.edge_index).cpu().numpy()
logger.info("Computed spatial embeddings.")

# -----------------------------------------------------------------------------
# Interaction extraction functions
# -----------------------------------------------------------------------------
def get_cell_neighbors_types_and_expr(adata, knn_adj, idx):
    nbr_idxs = knn_adj.getrow(idx).indices
    nbr_types = adata.obs['Cell_class'].iloc[nbr_idxs].values
    nbr_expr = adata.X[nbr_idxs]
    return nbr_idxs, nbr_types, nbr_expr

from sklearn.metrics.pairwise import cosine_similarity
def get_ligand_receptor_candidates_with_norms_and_proximity(adata, knn_adj, cell_idx, spatial_embeddings, expr_threshold=1):
    cell_type = adata.obs['Cell_class'].iloc[cell_idx]
    nbr_idxs = knn_adj.getrow(cell_idx).indices
    nbr_types = adata.obs['Cell_class'].iloc[nbr_idxs].values
    #
    # Source expression
    cell_expr = adata.X[cell_idx]
    if hasattr(cell_expr, "toarray"):
        cell_expr = cell_expr.toarray().flatten()
    elif hasattr(cell_expr, "A1"):
        cell_expr = cell_expr.A1
    else:
        cell_expr = np.asarray(cell_expr).flatten()
    #
    cell_genes = adata.var_names[np.where(cell_expr > expr_threshold)[0]]
    #
    interactions = []
    source_vec = spatial_embeddings[cell_idx]  # <-- define once
    source_norm = np.linalg.norm(source_vec)
    #
    for local_i, nbr_i in enumerate(nbr_idxs):
        nbr_expr_vec = adata.X[nbr_i]
        if hasattr(nbr_expr_vec, "toarray"):
            nbr_expr_vec = nbr_expr_vec.toarray().flatten()
        elif hasattr(nbr_expr_vec, "A1"):
            nbr_expr_vec = nbr_expr_vec.A1
        else:
            nbr_expr_vec = np.asarray(nbr_expr_vec).flatten()
            #
        nbr_genes = adata.var_names[nbr_expr_vec > expr_threshold]
        target_type = nbr_types[local_i]
        target_vec = spatial_embeddings[nbr_i]
        target_norm = np.linalg.norm(target_vec)
        #
        # Proximity metrics
        cosine_sim = cosine_similarity(
            source_vec.reshape(1, -1),
            target_vec.reshape(1, -1)
        )[0, 0]
        euclidean_dist = np.linalg.norm(source_vec - target_vec)
        #
        for lig in cell_genes:
            if lig in ligand_receptor_dict:
                for rec in ligand_receptor_dict[lig]:
                    if rec in nbr_genes:
                        interactions.append({
                            'source_cell': cell_idx,
                            'source_type': cell_type,
                            'target_cell': nbr_i,
                            'target_type': target_type,
                            'ligand': lig,
                            'receptor': rec,
                            'source_norm': source_norm,
                            'target_norm': target_norm,
                            'cosine_similarity': cosine_sim,
                            'euclidean_distance': euclidean_dist,
                        })
    return interactions

def stratified_sample_cells(adata, n_cells_total):
    # Get unique cell types
    cell_types = adata.obs['Cell_class'].unique()
    n_types = len(cell_types)
    
    # How many samples per cell type (integer division)
    n_per_type = max(1, n_cells_total // n_types)
    
    sampled_idxs = []
    for ctype in cell_types:
        # Indices for this cell type
        idxs = np.where(adata.obs['Cell_class'] == ctype)[0]
        # If not enough cells, sample all; else random sample n_per_type without replacement
        if len(idxs) <= n_per_type:
            sampled_idxs.extend(idxs.tolist())
        else:
            sampled_idxs.extend(np.random.choice(idxs, size=n_per_type, replace=False).tolist())
    
    return np.array(sampled_idxs)

def aggregate_interactions_with_norms(adata, knn_adj, spatial_embeddings, n_cells=100, expr_threshold=1):
    all_interactions = []
    #sampled_idxs = np.random.choice(adata.n_obs, size=n_cells, replace=False)
    sampled_idxs = stratified_sample_cells(adata, n_cells)
    for i in sampled_idxs:
        all_interactions.extend(get_ligand_receptor_candidates_with_norms_and_proximity(adata, knn_adj, i, spatial_embeddings, expr_threshold=expr_threshold))
    df_int = pd.DataFrame(all_interactions)
    #
    # Convert categorical columns to string to avoid grouping errors
    for col in ['source_type', 'target_type', 'ligand', 'receptor']:
        df_int[col] = df_int[col].astype(str)
    #
    # Aggregate count and average norms by source_type, target_type, ligand, receptor
    agg = df_int.groupby(['source_type', 'target_type', 'ligand', 'receptor']).agg(
        count=('ligand', 'size'),
        avg_source_norm=('source_norm', 'mean'),
        avg_target_norm=('target_norm', 'mean'),
        avg_cosine_similarity=('cosine_similarity', 'mean'),
    ).reset_index()
    #
    return agg

# -----------------------------------------------------------------------------
# Prompt builder using norm of embeddings
# -----------------------------------------------------------------------------
def build_cci_prompt_with_norms(row, marker_dict):
    source_type = row['source_type']
    target_type = row['target_type']
    ligand = row['ligand']
    receptor = row['receptor']
    count = row['count']
    avg_source_norm = row['avg_source_norm']
    avg_target_norm = row['avg_target_norm']
    avg_cosine_similarity = row['avg_cosine_similarity']
    #
    source_markers = marker_dict.get(source_type, [])
    target_markers = marker_dict.get(target_type, [])
    #
    return (
        f"Analyze the following potential cell–cell interaction in the Idiopathic Pulmonary Fibrosis (IPF) dataset.\n\n"
        
        f"--- CELL TYPE CONTEXT ---\n"
        f"Source cell type: {source_type}\n"
        f"  Representative marker genes (for identifying cell type only): {', '.join(source_markers)}\n"
        f"Target cell type: {target_type}\n"
        f"  Representative marker genes (for identifying cell type only): {', '.join(target_markers)}\n\n"
        
        f"--- LIGAND–RECEPTOR CANDIDATE ---\n"
        f"Ligand: {ligand} (expressed above threshold in the source cell)\n"
        f"Receptor: {receptor} (expressed above threshold in the target cell)\n"
        f"Expression threshold applied per cell, not by average per cell type.\n\n"
        
        f"--- SPATIAL OCCURRENCE ---\n"
        f"Observed in {count} distinct spatial neighborhoods (frequency of co-occurrence of ligand and receptor expression in nearby cells).\n"
        f"Average source cell embedding norm: {avg_source_norm:.2f} (reflecting position strength in learned spatial–expression space)\n"
        f"Average target cell embedding norm: {avg_target_norm:.2f} (same scale as above)\n"
        f"Higher norms may correspond to stronger local spatial signal.\n\n"
        f"Average cosine similarity: {avg_cosine_similarity:.3f} (1 = identical spatial context, 0 = unrelated)\n"
        f"Cosine similarity is the primary indicators of spatial proximity.\n\n"

        f"--- TASK ---\n"
        f"Based on the cell type context, ligand–receptor expression evidence, and spatial occurrence data, "
        f"decide whether this represents a plausible biological interaction in the IPF context. "
        f"Return a JSON object with the following keys:\n"
        f"- 'interaction'\n"
        f"- 'confidence': number between 0 and 1\n"
        f"- 'justification': short explanation grounded in the provided data and known biology."
    )

# -----------------------------------------------------------------------------
# Agent setup functions
# -----------------------------------------------------------------------------
def configure_agents():
    tools = []
    inf = Agent(
        role="Bioinformatics Researcher",
        goal="Analyze potential ligand-receptor interactions between cell types in Idiopathic Pulmonary Fibrosis (IPF) data and return JSON with keys 'interaction', 'confidence', 'justification'.",
        backstory=(
            "A junior computational biologist focused on cell-cell communication inference from spatial transcriptomics data, "
            "using marker genes, ligand-receptor databases, spatial neighborhood info, and embeddings."
        ),
        allow_delegation=False,
        tools=tools,
        llm=LLM(
            model="vertex_ai/gemini-2.5-flash",
            temperature=0.7,
            vertex_project=os.environ["GOOGLE_CLOUD_PROJECT"],
            vertex_location=os.environ["GOOGLE_CLOUD_LOCATION"],
        ),
        verbose=False,
        memory=False
    )
    rev = Agent(
        role="Senior Quality Assurance Bioinformatician",
        goal="Review a junior agent’s ligand-receptor interaction prediction and return JSON with a single key 'interaction'.",
        backstory=(
            "A senior scientist specializing in cell-cell interaction validation in spatial omics."
        ),
        allow_delegation=False,
        tools=tools,
        llm=LLM(
            model="vertex_ai/gemini-2.5-flash",
            temperature=0.7,
            vertex_project=os.environ["GOOGLE_CLOUD_PROJECT"],
            vertex_location=os.environ["GOOGLE_CLOUD_LOCATION"],
        ),
        verbose=False,
        memory=False
    )
    return inf, rev

inference_agent, review_agent = configure_agents()

@lru_cache(maxsize=10000)
def call_inference(prompt):
    for attempt in range(3):
        start = time.time()
        task = Task(description=prompt, expected_output="JSON with keys 'interaction','confidence','justification'", output_file="inf_cci.json", agent=inference_agent)
        try:
            raw = Crew(agents=[inference_agent], tasks=[task], process=Process.sequential).kickoff().raw
            logger.info(f"Inference call done in {time.time()-start:.2f}s")
            js = json.loads(re.sub(r"```json|```", "", raw).strip())
            if isinstance(js, list) and len(js) > 0 and isinstance(js[0], dict):
                js = js[0]
            return js
        except Exception as e:
            logger.warning(f"Inference attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return {"interaction": "None", "confidence": 0.0, "justification": ""}

def call_review(inf_js):
    prompt = f"Candidates: potential interactions\nInference JSON: {json.dumps(inf_js)}\nReturn JSON with single key 'interaction'."
    start = time.time()
    task = Task(description=prompt, expected_output="JSON with key 'interaction'", agent=review_agent)
    raw = Crew(agents=[review_agent], tasks=[task], process=Process.sequential).kickoff().raw
    logger.info(f"Review call done in {time.time()-start:.2f}s")
    try:
        rev = json.loads(re.sub(r"```json|```", "", raw).strip())
        inter = rev.get('interaction', 'None')
        return inter
    except:
        return 'None'

# -----------------------------------------------------------------------------
# Evaluation pipeline for cell-cell interactions
# -----------------------------------------------------------------------------
def evaluate_cci_pipeline_with_norms(n_samples=100, use_review=False, max_workers=8):
    logger.info(f"Aggregating ligand-receptor interactions on {n_samples} cells (with norms)...")
    agg_df = aggregate_interactions_with_norms(adata, knn_adjacency, spatial_embeddings, n_cells=n_samples)
    #
    records = []
    logger.info(f"Running inference on {len(agg_df)} candidate interactions...")
    #
    from concurrent.futures import ThreadPoolExecutor
    #
    def worker(row):
        prompt = build_cci_prompt_with_norms(row, marker_dict)
        inf_js = call_inference(prompt)
        conf = inf_js.get('confidence', 0.0)
        justification = inf_js.get('justification', "")
        interaction = inf_js.get('interaction', "None")
    #    
        if use_review:
            review_decision = call_review(inf_js)
            if review_decision and review_decision != 'None':
                interaction = review_decision
    #   
        return {
            'source_type': row['source_type'],
            'target_type': row['target_type'],
            'ligand': row['ligand'],
            'receptor': row['receptor'],
            'count': row['count'],
            'avg_source_norm': row['avg_source_norm'],
            'avg_target_norm': row['avg_target_norm'],
            'avg_cosine_similarity': row['avg_cosine_similarity'],
            'interaction': interaction,
            'confidence': conf,
            'justification': justification,
        }
    #
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, [row for _, row in agg_df.iterrows()]))
    #
    results_df = pd.DataFrame(results)
    logger.info(f"Completed inference on all candidates.")
    return results_df


# -----------------------------------------------------------------------------
# Example marker dictionary for prompt building
# -----------------------------------------------------------------------------
# Run differential expression test per cell type (groups by 'Cell_class')
sc.tl.rank_genes_groups(adata, groupby='Cell_class', method='t-test', n_genes=20)

# Extract top 10 marker genes per cell type into a dict
marker_dict = {}
groups = adata.obs['Cell_class'].unique()

for group in groups:
    # The names stored may be numeric strings (indices) - convert to int indices
    indices_str = adata.uns['rank_genes_groups']['names'][group][:10]
    # Convert string indices to integers
    indices = indices_str.astype(int)
    # Map indices to actual gene names from var_names
    top_genes = adata.var_names[indices]
    marker_dict[group] = list(top_genes)

# Inspect marker_dict
for cell_type, markers in marker_dict.items():
    print(f"{cell_type}: {markers}")

# -----------------------------------------------------------------------------
# Main execution example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting evaluation without review step...")
    num_classes = adata.obs['Cell_class'].nunique()
    res_single = evaluate_cci_pipeline_with_norms(n_samples=100 * num_classes, use_review=False, max_workers=8)
    res_single.to_csv("/../cci_inference_results_no_review_0_withDistance.csv", index=False) 

    res_single_filtered = res_single[res_single["confidence"] > 0.7].copy()
    res_single.to_csv("/../cci_inference_results_no_review_0_withDistance_0.7.csv", index=False)


    logger.info("Starting evaluation with review step...")
    res_dual = evaluate_cci_pipeline_with_norms(n_samples=100 * num_classes, use_review=True, max_workers=8)
    res_dual.to_csv("cci_inference_results_with_review.csv", index=False)

    res_dual.to_csv("/../cci_inference_results_no_review_0_withDistance_2A.csv", index=False) 
    res_dual_filtered = res_dual[res_dual["confidence"] > 0.7].copy()
    res_dual_filtered.to_csv("/../cci_inference_results_no_review_0_withDistance_2A_0.7.csv", index=False)

    logger.info("Evaluation completed.")


# The binning shows a monotonic trend: more occurrences in spatial neighborhoods generally correspond to higher confidence
res_single_filtered['count_bin'] = pd.cut(res_single_filtered['count'], bins=[0,1,2,5,10,20,50,100, np.inf])
bin_means = res_single_filtered.groupby('count_bin')['confidence'].mean()
print(bin_means)

correlation = res_single_filtered['avg_source_norm'].corr(res_single_filtered['avg_target_norm'])
print(f"Correlation between avg_source_norm and avg_target_norm: {correlation}")

correlation = res_single_filtered['count'].corr(res_single_filtered['confidence'])
print(f"Correlation between count and confidence: {correlation}")

correlation = res_single_filtered['avg_cosine_similarity'].corr(res_single_filtered['confidence'])
print(f"Correlation between avg_cosine_similarity and confidence: {correlation}")


# -----------------------------------------------------------------------------
# Evaluation cell2cell
# -----------------------------------------------------------------------------

# Contingency matrix
# -----------------

# Load gold standard
gold = pd.read_csv(
    "/../IPF_gold_standard.txt",
    sep="\t",
    encoding="utf-16le"
)

# Make sure column names match between gold and matched_res
gold.columns = ["source_type", "target_type", "ligand", "receptor"]

# Inner join to find matches
matched_in_gold = pd.merge(
    res_single_filtered,
    gold,
    on=["source_type", "target_type", "ligand", "receptor"],
    how="inner"
)

# Count matches
num_matches = len(matched_in_gold)
print(f"Matches with gold standard: {num_matches}")

# Unique source and target cell types in your dataset
cell_types = pd.concat([res_single_filtered["source_type"], res_single_filtered["target_type"]]).unique()

# All ligands and receptors observed in dataset
ligands = pd.concat([res_single_filtered["ligand"], gold["ligand"]]).unique()
receptors = pd.concat([res_single_filtered["receptor"], gold["receptor"]]).unique()

# Cartesian product to generate all possible pairs
allpairs = pd.DataFrame(list(itertools.product(cell_types, cell_types, ligands, receptors)),
                        columns=["source_type","target_type","ligand","receptor"])

print("Total possible pairs:", len(allpairs))


def run_CellPhoneDB_SSP(posi, allpairs, gold):
    # Only keep relevant columns
    posi = posi[["source_type", "target_type", "ligand", "receptor"]].copy()
    
    # Create interaction IDs for both directions
    posi["ID1"] = posi["source_type"] + "-" + posi["target_type"] + "-" + posi["ligand"] + "-" + posi["receptor"]
    posi["ID2"] = posi["target_type"] + "-" + posi["source_type"] + "-" + posi["receptor"] + "-" + posi["ligand"]
    
    gold = gold[["source_type", "target_type", "ligand", "receptor"]].copy()
    gold["ID"] = gold["source_type"] + "-" + gold["target_type"] + "-" + gold["ligand"] + "-" + gold["receptor"]
    
    # True Positives
    TP = ((posi["ID1"].isin(gold["ID"])) | (posi["ID2"].isin(gold["ID"]))).sum()
    
    # False Positives
    FP = len(posi) - TP
    
    # False Negatives
    FN = len(gold) - TP
    
    # True Negatives
    TN = len(allpairs) - (TP + FP + FN)
    
    # Metrics
    acc = (TP + TN) / len(allpairs)
    pre = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    sst = TP / (TP + FN) if (TP + FN) > 0 else np.nan  # sensitivity
    spc = TN / (FP + TN) if (FP + TN) > 0 else np.nan  # specificity
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan
    mcc = ((TP * TN - FP * FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) if all(x>0 for x in [TP+FP, TP+FN, TN+FP, TN+FN]) else np.nan
    npv = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    
    SSP = {
        "Precision": pre,
        "Sensitivity": sst,
        "Specificity": spc,
        "F1score": f1,
        "MCC": mcc,
        "NPV": npv,
        "Accuracy": acc
    }
    
    return SSP


# Assuming res_single_filtered is your LLM-filtered results (confidence > 0.7)
# allpairs can be all possible source-target-ligand-receptor combinations
metrics = run_CellPhoneDB_SSP(res_single_filtered, allpairs, gold)
metrics = {k: round(float(v), 7) for k, v in metrics.items()}
print(metrics)


# Per-cell gene expression distribution
# -----------------

cell_idx = 0  # or any cell index you want to inspect
cell_expr = adata.X[cell_idx]

# If sparse matrix, convert to dense
if hasattr(cell_expr, "toarray"):
    cell_expr = cell_expr.toarray().flatten()
elif hasattr(cell_expr, "A1"):
    cell_expr = cell_expr.A1
else:
    cell_expr = np.asarray(cell_expr).flatten()

plt.hist(cell_expr, bins=100, log=True)
plt.xlabel("Expression value")
plt.ylabel("Frequency (log scale)")
plt.title(f"Expression distribution for cell index {cell_idx}")
plt.show()

# Calculate fraction of genes expressed above threshold candidates
for thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 3.0, 4.0, 5.0, 7.5, 10]:
    fraction = np.mean(cell_expr > thresh)
    print(f"Fraction of genes with expr > {thresh}: {fraction:.4f}")


# ============================================================
# Frozen priors
# Train RF on LLM outputs, test on original LLM rows,
# evaluate on 10 new LR input sets,
# evaluate on 10 biologically-derived FALSE LR sets.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------------------------------------
# 0. ASSUMPTIONS
#    adata, knn_adjacency, spatial_embeddings, 
#    aggregate_interactions_with_norms()
#    already exist in memory.
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1. LOAD LLM OUTPUT (TRAINING DATA)
# ------------------------------------------------------------

df_llm = pd.read_csv(
    "/../cci_inference_results_no_review_0_withDistance_0.7.csv"
)

df_llm["label"] = df_llm["interaction"].isin(["True", "Plausible"]).astype(int)

categorical = ["source_type", "target_type", "ligand", "receptor"]
numeric = ["count", "avg_source_norm", "avg_target_norm", "avg_cosine_similarity"]

print("Training data:", df_llm.shape)
print("LLM % positive:", df_llm["label"].mean() * 100)


# ------------------------------------------------------------
# 1b. LOAD CURATED LR DATABASE
# ------------------------------------------------------------

lr_pairs = pd.read_csv(
    "/../combined_ligand_receptor_pairs.csv"
)

known_lr_set = set(zip(lr_pairs["ligand_gene"], lr_pairs["receptor_gene"]))

all_ligands = sorted(df_llm["ligand"].unique())
all_receptors = sorted(df_llm["receptor"].unique())


# ------------------------------------------------------------
# 1c. PRECOMPUTE EXPRESSION PRESENCE PER CELL TYPE
# ------------------------------------------------------------

expr_threshold = 1.0
cell_types = adata.obs["Cell_class"].unique()

gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
genes_of_interest = [g for g in (set(all_ligands) | set(all_receptors)) if g in gene_to_idx]

expr_present = {}

print("\nPrecomputing expression presence across cell types...")
for ct in cell_types:
    mask = (adata.obs["Cell_class"] == ct).values
    X_sub = adata.X[mask, :]
#
    for g in genes_of_interest:
        gi = gene_to_idx[g]
        col = X_sub[:, gi]
#
        if hasattr(col, "A1"):
            vals = col.A1
        elif hasattr(col, "toarray"):
            vals = col.toarray().ravel()
        else:
            vals = np.asarray(col).ravel()
#
        expr_present[(ct, g)] = (vals > expr_threshold).any()

print("✓ Expression matrix ready.")


# ------------------------------------------------------------
# 2. ONE-HOT ENCODER (TRAIN ONLY)
# ------------------------------------------------------------

enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
enc.fit(df_llm[categorical])

X_train_full = np.hstack([
    df_llm[numeric].values,
    enc.transform(df_llm[categorical])
])

y_train = df_llm["label"]

print("Training feature size:", X_train_full.shape)


# ------------------------------------------------------------
# 3. TRAIN RANDOM FOREST
# ------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_full, y_train)

print("\nRF performance on training (LLM) data:")
print(classification_report(y_train, rf.predict(X_train_full)))


# ------------------------------------------------------------
# 4. TEST ON ORIGINAL LLM ROWS
# ------------------------------------------------------------

X_test_full = np.hstack([
    df_llm[numeric].values,
    enc.transform(df_llm[categorical])
])

df_llm["rf_pred"] = rf.predict(X_test_full)
df_llm["rf_prob"] = rf.predict_proba(X_test_full)[:, 1]

print("\nRF % positive on LLM rows:", df_llm["rf_pred"].mean() * 100)
df_llm.to_csv("/../rf_on_llm_rows.csv", index=False)


# ------------------------------------------------------------
# 5. GENERATE 10 NEW LR DATASETS (REAL POSITIVES)
# ------------------------------------------------------------

def generate_lr_dataset(seed):
    np.random.seed(seed)
    return aggregate_interactions_with_norms(
        adata, knn_adjacency, spatial_embeddings,
        n_cells=100 * adata.obs["Cell_class"].nunique(),
        expr_threshold=1
    )

print("\n=== GENERATING 10 NEW POSITIVE-LIKE DATASETS ===\n")

new_datasets = []
pos_summary = []

for i in range(10):
    seed = 1000 + i
    agg_new = generate_lr_dataset(seed)
#
    X_full = np.hstack([
        agg_new[numeric].values,
        enc.transform(agg_new[categorical])
    ])
#
    agg_new["rf_pred"] = rf.predict(X_full)
    agg_new["rf_prob"] = rf.predict_proba(X_full)[:, 1]
#
    pos_rate = agg_new["rf_pred"].mean() * 100
    print(f"Dataset {i+1}: {pos_rate:.2f}% positives")
#
    pos_summary.append(pos_rate)
    new_datasets.append(agg_new)

pd.Series(pos_summary).to_csv("/../rf_newdatasets_posrate.csv", index=False)


# ------------------------------------------------------------
# 6. BIOLOGICALLY-INFORMED FALSE EXAMPLES FROM ADATA
# ------------------------------------------------------------

def generate_false_examples(
    agg_df, ligand_list, receptor_list,
    expr_present, known_lr_set,
    n_examples=10, seed=0
):
    rng = np.random.default_rng(seed)
    false_rows = []
#
    src_types = agg_df["source_type"].unique()
    tgt_types = agg_df["target_type"].unique()
#
    for _ in range(n_examples):
        s = rng.choice(src_types)
        t = rng.choice(tgt_types)
#
        lig_candidates = [L for L in ligand_list if not expr_present.get((s, L), True)]
        rec_candidates = [R for R in receptor_list if not expr_present.get((t, R), True)]
#
        if len(lig_candidates) == 0 or len(rec_candidates) == 0:
            continue
#
        # ensure LR not in curated DB
        for _ in range(30):
            L = rng.choice(lig_candidates)
            R = rng.choice(rec_candidates)
            if (L, R) not in known_lr_set:
                break
#
        false_rows.append({
            "source_type": s,
            "target_type": t,
            "ligand": L,
            "receptor": R,
            "count": 0,
            "avg_source_norm": rng.uniform(0, 3),
            "avg_target_norm": rng.uniform(0, 3),
            "avg_cosine_similarity": rng.uniform(-1, -0.1),
            "true_label": 0
        })
#
    return pd.DataFrame(false_rows)


# ============================================================
# 7) EVALUATE RF ON STRONG FALSE EXAMPLES
# ============================================================

false_results = []   # STORE NEGATIVE RESULTS FOR METADATA

print("\n=== RF performance on BIOLOGICALLY-INFORMED FALSE examples ===\n")

for i, df_new in enumerate(new_datasets):
    print(f"\n--- Dataset {i+1} ---")
#
    df_false = generate_false_examples_strong(
        df_new,
        all_ligands,
        all_receptors,
        expr_present,
        known_lr_set,
        n_examples=10,
        seed=2000 + i
    )
#
    if df_false.empty:
        print(f"Dataset {i+1}: no strong false LR generated.")
        false_results.append(pd.DataFrame())   # keep placeholder
        continue
#
    # Prepare features
    X_test_cat = df_false[categorical]
    X_test_num = df_false[numeric]
#
    X_test_full = np.hstack([
        X_test_num.values,
        enc.transform(X_test_cat)
    ])
#
    df_false["rf_pred"] = rf.predict(X_test_full)
    df_false["rf_prob"] = rf.predict_proba(X_test_full)[:, 1]
#
    n_fp = int(df_false["rf_pred"].sum())
    total = len(df_false)
#
    print(f"Dataset {i+1}: FALSE predicted TRUE = {n_fp}/{total}")
#
    # store negatives
    false_results.append(df_false)
#
    # save negative examples file
    df_false.to_csv(f"rf_negatives_newset_{i+1}.csv", index=False)


# ============================================================
# GLOBAL METADATA
# ============================================================

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1) LLM TRAINING DISTRIBUTION
# ------------------------------------------------------------
llm_true_pct = df_llm["label"].mean() * 100
llm_false_pct = 100 - llm_true_pct

# ------------------------------------------------------------
# 2) RF 75/25 SPLIT PERFORMANCE
# ------------------------------------------------------------
X = X_train_full
y = y_train

X_train, X_test, y_train_split, y_test_split = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

rf_split = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_split.fit(X_train, y_train_split)

train_acc = rf_split.score(X_train, y_train_split) * 100
test_acc = rf_split.score(X_test, y_test_split) * 100

train_true_pct = y_train_split.mean() * 100
test_true_pct = y_test_split.mean() * 100

# ------------------------------------------------------------
# 3) GOLD-STANDARD BIOLOGY DISTRIBUTION
# ------------------------------------------------------------
gold_labels = df_llm.apply(
    lambda row: int((row["ligand"], row["receptor"]) in known_lr_set),
    axis=1
)
gold_true_pct = gold_labels.mean() * 100
gold_false_pct = 100 - gold_true_pct


# ------------------------------------------------------------
# SAVE GLOBAL METADATA TABLE
# ------------------------------------------------------------
global_metadata = pd.DataFrame([{
    "llm_true_pct": llm_true_pct,
    "llm_false_pct": llm_false_pct,
    "rf_train_accuracy_pct": train_acc,
    "rf_test_accuracy_pct": test_acc,
    "train_true_pct": train_true_pct,
    "test_true_pct": test_true_pct,
    "gold_true_pct": gold_true_pct,
    "gold_false_pct": gold_false_pct
}])

global_metadata.to_csv("/../rf_global_metadata.csv", index=False)

print("\n=== SAVED GLOBAL METADATA → rf_global_metadata.csv ===")
print(global_metadata)


# ============================================================
# DATASET-LEVEL METADATA
# ============================================================

dataset_records = []

for i in range(10):
    dataset_id = i + 1
    seed = 1000 + i
#
    # --- positive dataset ---
    agg_new = new_datasets[i]
    n_rows = len(agg_new)
    pos_rate = agg_new["rf_pred"].mean() * 100
#
    # --- negative dataset ---
    df_false = false_results[i]
    n_neg = len(df_false)
    neg_fp = int(df_false["rf_pred"].sum()) if n_neg > 0 else 0
    neg_fp_rate = (neg_fp / n_neg * 100) if n_neg > 0 else np.nan
#
    dataset_records.append({
        "dataset_id": dataset_id,
        "seed": seed,
        "n_rows": n_rows,
        "positive_rate_percent": pos_rate,
        "n_negatives_generated": n_neg,
        "negatives_pred_true": neg_fp,
        "negative_false_positive_rate": neg_fp_rate
    })

dataset_metadata_df = pd.DataFrame(dataset_records)
dataset_metadata_df.to_csv("/../rf_dataset_metadata.csv", index=False)

print("\n=== SAVED DATASET-LEVEL METADATA → rf_dataset_metadata.csv ===")
print(dataset_metadata_df)
