# About Simpatico 
Simpatico is a graph neural network for producing high-dimensional embeddings of atoms in proteins and small molecules. Atomic representations produced by Simpatico are co-located in embedding space according to their interaction potential. This allows users to perform rapid virtual screening over extremely large datasets. See our [paper](https://www.biorxiv.org/content/10.1101/2025.06.08.658499v2) for further details.

**This repo is being actively updated. If you encounter a problem, please download the latest version first and see if this solves the issue. If the problem persists, contact the authors.**

## Installation

<details closed>
<summary><strong>Installing simpatico on your system</strong></summary>
Simpatico depends on several GPU-based libraries such as PyTorch that are sensitive to your computing environment (e.g., CUDA version and GPU availability). These dependencies include PyTorch, PyG, and Faiss.

The following sequence of commands will work for most users. This procedure is assembled from installation instructions provided by the respective libraries:

* PyTorch – [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* PyG – [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* Faiss – [https://pypi.org/project/faiss-gpu-cu12/](https://pypi.org/project/faiss-gpu-cu12/)

### 1. Installing PyTorch
```bash
pip install torch
```

Verify your PyTorch and CUDA versions:

```bash
python -c "import torch; print(torch.__version__)"
```

This will produce a value like:

```bash
2.7.0+cu126
```

If you don’t see a `cu{NUMBER}` value, try:

```bash
python -c "import torch; print(torch.version.cuda)"
```

### 2. Installing PyG

First, install the base library of PyTorch Geometric.
```bash
pip install torch_geometric
```

Then, install the correct PyG packages according to the version numbers from the previous command:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

**Important:** Replace `torch-2.7.0+cu126.html` with your specific PyTorch and CUDA version.

### 3. Installing Faiss

Install the Faiss GPU library matching your CUDA version:

```bash
pip install faiss-gpu-cu12
```

Again, adjust `cu12` to correspond to your CUDA version if needed.

### 4. Installing simpatico

With all dependencies installed, you can now install simpatico itself:

```bash
git clone https://github.com/TravisWheelerLab/Simpatico.git
pip install Simpatico
```

</details>

## Usage
For demonstration purposes, protein and molecule structures have been sourced from a test-screening sample from the DUDE dataset (aa2ar), located in `Simpatico/examples/aa2ar_screen`. All commands described below are run from the root `Simpatico` directory.

<details closed>
<summary><strong>Virtual screening a small-molecule database for a protein target</strong></summary>
Virtual screening is performed by using protein pocket embeddings to query a Faiss vector database of small-molecule atom embeddings. An aggregation procedure over the atom embeddings produces a score for every molecule containing an atom observed during the nearest neighbors search process. 

To run a query, prepare a CSV file specifying which embedding files to use as queries (e.g., protein pockets) and which to use as the vector database (e.g., candidate molecules). The following csv file is used to run our example screening: 

**examples/aa2ar_screen/aa2ar\_screen\_1.csv**
```
q,examples/aa2ar_screen/receptor.pdb,examples/aa2ar_screen/crystal_ligand.mol2
d,examples/aa2ar_screen/molecule_library_1.ism
d,examples/aa2ar_screen/molecule_library_2.ism
```

For each line, the first column specifies the data type with a single character, either a `q` (query) or `d` (database). This will always be followed by a second column specifying the path to the molecular data file.

When specifying the protein query, we must include an additional third column to specify the location of the protein target pocket. This will usually be a 3D small molecule file (e.g. `.sdf` or `.mol2`) from which 3D coordinates may be extracted. A ligand structure docked in the target pocket is ideal for this purpose.  

To run the query:

### Command Usage

```bash
simpatico query <input_file> <output_file>
```

### Example

```bash
simpatico query examples/aa2ar_screen/aa2ar_screen_1.csv examples/aa2ar_screen/results/
```

This will generate two result files, saved to the `examples/aa2ar_screen/results/` directory, one per small-molecule database specified in the input file. `molecule_library_1_query-results.csv` should look something like this:

```
1,11,37.93132781982422
1,209,34.20753860473633
1,211,25.427879333496094
1,145,24.6605281829834
1,379,24.286479949951172
...
```

Each row of the results `.csv` has three columns: The query index, small-molecule index, and the score. In effect, for every query included in the input file, each non-zero scoring small molecule is listed in order of score, from highest (best) scoring to lowest score. In our example case, molecules 1-482 from `molecule_library_1.ism` are known actives, and therefore occupy an outsized proportion of high scoring rows.

</details>

<details closed>
<summary><strong>Generating protein and small molecule atom embeddings</strong></summary>

The previous screening example was quite slow. This is because for each small molecule library, we generated graphs, ran inference, and then finally performed the search-based screening process. In practice, it may be more efficient to generate small molecule embeddings ahead of time. Then, any number of queries may be used for rapid downstream screening.

To generate small molecule embeddings, we just need a list of the molecule libraries:

**examples/aa2ar_screen/mol_lib_embed.txt**

```
examples/aa2ar_screen/molecule_library_1.ism
examples/aa2ar_screen/molecule_library_2.ism
```

Run the command as follows:

### Command Usage

```bash
simpatico query <input_file> <output_dir> [-m|-p]
```

### Example

```bash
simpatico eval examples/aa2ar_screen/mol_lib_embed.txt examples/aa2ar_screen/embeddings -m
```
Note the `-m` flag to specify that we are converting a batch of small-molecules. For proteins, we would include `-p`.

For each specified molecule, a `.pyg` file containing a [batch of graphs](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.data.Batch.html) will be generated. Embedding values are stored in `graph.x`. Using the `.pyg` embedding files as the molecular library (instead of `.ism` files) will result in dramatically faster screen times. 
</details>

## Training
<details closed>
<summary><strong>Training and updating model weights</strong></summary>
Simpatico is trained on structural data of protein-ligand complexes. Each training sample consists of one protein structure and one ligand structure, which together make up the bound protein-ligand complex. 

Input for a simpatico training run is stored in the json format. From our example:

**examples/train_example/train_example.json**
```bash
{
    "train_handle": "simpatico_train_example",
    "data_file": "examples/train_example/PDBBIND_sample.pkl",
    "validation_file": "examples/train_example/example_validation.txt",
    "holdout_file": "examples/train_example/example_holdout.txt",
    "output_dir": "examples/train_example/",
    "batch_size": 16,
    "epochs": 1000,
    "learning_rate": 0.0001
    "weight_checkpoint_interval": 5
}
```
`train_handle` specifies a unique string to associate with the weight and log outputs. `data_file` must point to a pickle (`.pkl`) file containing a python list-of-lists. Each list-item in the list contains a protein graph in index 0, and the graph of its bound ligand partner in index 1. We have stored a (very) small sample of graphs from sourced from the PDBBind dataset in `examples/train_example/PDBBIND_sample.pkl`. `validation_file` and `holdout_file` should point to text files that list per-line a substring that may be found in the `.name` attribute of our training graphs (in our case, this is PDB IDs). If the substring is observed, the corresponding sample will be used withheld from the training data and used in the validation set (if listed in the `validation_file`) or simply witheld from training (if listed in the `holdout_file`). `output_dir` specifies where files generated during training run (weights, log files) should be sent. 
### Command Usage

```bash
simpatico train <input_file> 
```

### Example

```bash
simpatico train examples/train_example/train_example.json
```  
</details closed>


## Authors

**Jeremiah Gaiser**
School of Information
University of Arizona
Tucson, AZ 85721
[jgaiser@arizona.edu](mailto:jgaiser@arizona.edu)

**Travis J. Wheeler**
College of Pharmacy
University of Arizona
Tucson, AZ 85721
[twheeler@arizona.edu](mailto:twheeler@arizona.edu)
