# Experiments folder


This folder contains some of the experiments I carried out. The main objective of the project is stated in the [report](https://drive.google.com/file/d/14VLPiCDF7ntZ2c0Nc70Fu9qHq1o7pIpS/view?usp=sharing). Here, I provide a brief description of what each notebook contains.

- [0_pfam.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/0_pfam.ipynb): I did some experiments with Pfam using NetworkX.

- [1_blastp.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/1_blastp.ipynb): Retrieve the sequence using the API.

- [2_extract_protein_from_xml.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/2_extract_protein_from_xml.ipynb): Extract the protein sequences from the BLAST XML output file.

- [3_tokenization_axa.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/3_tokenization_axa.ipynb): Tokenize and pad the sequances using a fixed vocabulary.

- [4_byte_pair_encoding.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/4_byte_pair_encoding.ipynb): Extract the sequences, train the SentEval model and produce the BPE encoding of the sequences.

- [5_esm_sequence_embedding.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/5_esm_sequence_embedding.ipynb): Extract the latent representations from the ESM2 model for each sequence.

- [6_human_gene_go_extraction.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/6_human_gene_go_extraction.ipynb): Retrieve the GO terms, tokenize them with a fixed vocabulary, pad them and store for future experiments.

- [7_load_goterm.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/7_load_goterm.ipynb): Visualization of the retrieved GO terms.

- [8_protBERT_precomputed_embeddings.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/8_protBERT_precomputed_embeddings.ipynb): Extraction of the precomputed embedding.

- [9_architecture_1_go.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/9_architecture_1_go.ipynb): 🚧Architecture under development for future investigation🚧 The idea is to combine multiple sources of information to get new sequences. I discussed the topic in further detail in the [report](https://drive.google.com/file/d/14VLPiCDF7ntZ2c0Nc70Fu9qHq1o7pIpS/view?usp=sharing).

- [9_architecture_2_emb-per-residue.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/9_architecture_2_emb-per-residue.ipynb): The first implementation of the cross-dimensions weighting block. Then, has been extended to the different backbones: token with fixed vocabulary, token with learnable vocabulary and ESM2 embedding.

- [10_retrive_the_entry_from_the_accession.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/10_retrive_the_entry_from_the_accession.ipynb): Retrieve Swiss-Prot entry from the accession number.

- [11_preprocess_esm_embedding.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/11_preprocess_esm_embedding.ipynb): Pipeline to preprocess the ESM2 data.

- [11_LinWeighingProteinEmbedding_ESM2.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/11_LinWeighingProteinEmbedding_ESM2.ipynb): The cross-dimensions weighting block is applied to perform classification using a simple MLP over the extracted embedding. This implementation has as a backbone the ESM2 embedding.

- [11.1_latent_space_visualization_TSNE.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/11.1_latent_space_visualization_TSNE.ipynb): Plot the latent embedding found training the architecture on the ESM2 embedding into a 2D space.

- [12_LinWeighingProteinEmbedding_axa.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/12_LinWeighingProteinEmbedding_axa.ipynb): The cross-dimensions weighting block is applied to perform classification using a simple MLP over the extracted embedding. This implementation has as a backbone the token with fixed vocabulary.

- [12.1_latent_space_visualization_TSNE_token.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/12.1_latent_space_visualization_TSNE_token.ipynb): Plot the latent embedding found training the architecture on the token with fixed vocabulary into a 2D space.

- [13_LinWeighingProteinEmbedding_axa_nn_EMBEDDING.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/13_LinWeighingProteinEmbedding_axa_nn_EMBEDDING.ipynb): The cross-dimensions weighting block is applied to perform classification using a simple MLP over the extracted embedding. This implementation has as a backbone the token with learnable vocabulary.

- [13.1_latent_space_visualization_TSNE_token_nn_EMBEDDING.ipynb](https://github.com/r1cc4r2o/UHC/blob/main/experiments/13.1_latent_space_visualization_TSNE_token_nn_EMBEDDING.ipynb): Plot the latent embedding found training the architecture on the token with learnable vocabulary into a 2D space.

- [module.py](https://github.com/r1cc4r2o/UHC/blob/main/experiments/module.py): Here, you can find the implemented module that you can directly import for further research.

- [axa_architecture.py](https://github.com/r1cc4r2o/UHC/blob/main/experiments/axa_architecture.py): Here, the implementation of the architecture with the token with fixed vocabulary.

- [axa_learnable_architecture.py](https://github.com/r1cc4r2o/UHC/blob/main/experiments/axa_learnable_architecture.py): Here, the implementation of the architecture with the token with learnable vocabulary.

- [esm_architecture.py](https://github.com/r1cc4r2o/UHC/blob/main/experiments/esm_architecture.py): Here, the implementation of the architecture with the ESM2 embedding.


REMINDER: I released the dataset in a gdrive folder here is the [link](https://drive.google.com/drive/folders/1IRZZGuC8f9lrTxA5k1fG8uOUA58Tkkym?usp=sharing). The structure of the folder has been discussed on the [front page](https://github.com/r1cc4r2o/UHC/tree/main#Dataset) under the sub-title Dataset.
