# GraphMethySite
# Example of lysine

# Data proprecess
# 1/ Download the original datasets from the website
# 2/ Run Alphafold2 to Map the PDB file format.
# After doing these, proprecess dataset first.
/data_proprecess/process_data.ipynb

# Train the Word2Vec model
/feature_extract_model/produce_word2vec.py

# After obtaining proprecessed data and training Word2Vec model, you need to construct the graphs.
# 1/ Calculate the adjacency matrix of graphs
# 2/ feature extracted matrix 
/data_proprecess/calculate_Adj.ipynb

# Run the model
/model_code/Runmodel.py: Run /model_code/Runmodel.py -h 10 -d 3 -t 'K' -cuda '2'

# BO is applied to select hyperparameter including hops and distance (Example of EI)
/model_code/Runmodel_BO_EI.py
