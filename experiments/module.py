### IMPORT
########################################################################################################
########################################################################################################

import pandas as pd
import matplotlib.pyplot as plt

from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import SeqIO, Entrez
from io import StringIO


import glob
import os

import torch
import torch.nn as nn

import numpy as np

import pickle

# !pip install git+https://github.com/facebookresearch/esm.git
import esm

# !pip install sentencepiece
import sentencepiece as spm

import multiprocessing
from Bio import Entrez, SeqIO
from tqdm import tqdm
import pandas as pd
from genes_ncbi_mus_musculus_proteincoding import GENEID2NT

import torch.nn.functional as F
import pytorch_lightning as pl



########################################################################################################
########################################################################################################



########################################################################################################

def get_protein_id(protein_name, mail: str = 'riccardo.tedoldi@studenti.unitn.it'):
    """ Get the NCBI Protein accession ID for a given protein name.

    Args:
        protein_name (str): The name of the protein.
        mail (str): The email address to use for the Entrez API.

    Returns:
        str: The NCBI Protein accession ID for the given protein name.

    """
    # Set up the Entrez API
    Entrez.email = mail
    db = "protein"

    # Send a search request to the NCBI Protein database
    handle = Entrez.esearch(db=db, term=protein_name)

    # Parse the search results to extract the NCBI Protein accession ID
    record = Entrez.read(handle)
    if int(record["Count"]) > 0:
        protein_id = record["IdList"][0]
        print(f"The NCBI Protein accession ID for {protein_name} is {protein_id}")
        handle.close()
        return protein_id
    else:
        handle.close()
        raise ValueError(f"No results found for {protein_name}")
    
########################################################################################################

def fetch_protein(proteins_name, protein_id, mail: str = 'riccardo.tedoldi@studenti.unitn.it'):
    """ Fetch a protein sequence from NCBI Protein database and save it in FASTA format

    Args:
        proteins_name (str): name of the protein
        protein_id (str): NCBI Protein accession ID
        mail (str): email address to use for the Entrez API

    Returns:
        None

    Save the protein sequence in FASTA format in the current working directory
    
    """

    # Set up the Entrez API
    Entrez.email = mail
    db = "protein"
    path = './proteins/'

    #Here, we set up a temporary handle with our downloaded sequence in fasta format
    temp = Entrez.efetch(db=db,rettype="fasta",id=protein_id)

    #Reading the sequence information as a string in fasta format
    seq = SeqIO.read(temp, format="fasta")

    #Creating a fasta file to write our downloaded sequence
    seq_out = open(f"{path}{proteins_name}.{protein_id}.fasta",'w')

    #Writing the sequence record in fasta format
    SeqIO.write(seq,seq_out,"fasta")

    #Closing both the temp handle and the FASTA file
    temp.close()
    seq_out.close()

########################################################################################################

def blastp(protein_name, db = 'nr'):
    """ Perform a blastp search for a given protein.

    Args:
        protein_name (str): The name of the protein.
        db (str): The database to use for the blastp search.

    Returns:
        None

    Save the blastp results in XML format in the folder 
    proteins/protein_name/blast_results_protein_name.db.xml

    """
    database = db

    path_dir = glob.glob('./proteins/*.fasta')

    for path in path_dir:

        # get the protein name
        protein_name = path.split('.')[1].split('/')[2]

        # create a folder for each protein if not already exists
        if not os.path.isdir(f'./proteins/{protein_name}'):
            os.mkdir(f'./proteins/{protein_name}')

        out_blast = f'./proteins/{protein_name}/'

        # read the fasta file
        seq = SeqIO.read(path, format='fasta')

        print(f"Performing blastp search for {protein_name} in {database} database")

        result_handle = NCBIWWW.qblast("blastp", database, seq.seq)

        print(f"Saving the results for {protein_name} in {database} database")

        # Save the results to a file
        with open(f"{out_blast}blast_results{protein_name}.{database}.xml", "w") as out_handle:
            out_handle.write(result_handle.read())

        print(f"Results saved for {protein_name} in {database} database")
        print()
            
        # Close the handle
        result_handle.close()



########################################################################################################

def from_xml_to_protein_sequences():
    """Extract the protein sequences from the BLAST XML output file and write to FASTA file.

    Args:
        None

    Returns:    
        None
    
    """

    # Open the BLAST XML output file
    blast_xml_files = glob.glob("./proteins/*/*-blastp.xml")

    for blast_xml_file in blast_xml_files:

        # Extract the protein name from the file path
        prot = blast_xml_file.split("/")[2]

        # Parse the BLAST output file
        with open(blast_xml_file) as blast_xml:
            blast_record = NCBIXML.read(blast_xml)

        # Extract the sequences from the BLAST output file and write to FASTA file
        sequences = []
        sequences_pck = []

        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:

                # Format the sbjct sequence as a FASTA record
                accession = alignment.hit_id.split("|")[1]
                sbjct_record = ">{} {}_{}\n{}\n".format(accession,alignment.hit_def, hsp.sbjct_start, hsp.sbjct)

                # Parse the FASTA record using SeqIO.read()
                seq_record = SeqIO.read(StringIO(sbjct_record), "fasta")
                sequences.append(seq_record)
                sequences_pck.append((accession, seq_record.seq))

                # Write the sequences to a FASTA file
                SeqIO.write(seq_record, f"./proteins/{prot}/{accession}.fasta", "fasta")

        with open(f"./proteins/{prot}/sequences/sequences_pck.pkl", "wb") as f:
            pickle.dump(dict(sequences_pck), f)
        # Write the sequences to a FASTA file
        SeqIO.write(sequences, f"./proteins/{prot}/sequences.fasta", "fasta")


########################################################################################################

def from_xml_to_txt_protein_sequences():
    """Extract the protein sequences from the BLAST XML output them 
    i a txt file.

    Args:
        None

    Returns:    
        None
    
    """

    # Open the BLAST XML output file
    blast_xml_files = glob.glob("./proteins/*/*-blastp.xml")

    for blast_xml_file in blast_xml_files:

        # Extract the protein name from the file path
        prot = blast_xml_file.split("/")[2]

        # Parse the BLAST output file
        with open(blast_xml_file) as blast_xml:
            blast_record = NCBIXML.read(blast_xml)

        # Extract the sequences from the BLAST output file and write to FASTA file
        sequences = []

        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:

                # Format the sbjct sequence as a FASTA record
                accession = alignment.hit_id.split("|")[1]
                sbjct_record = ">{} {}_{}\n{}\n".format(accession,alignment.hit_def, hsp.sbjct_start, hsp.sbjct)

                # Parse the FASTA record using SeqIO.read()
                seq_record = SeqIO.read(StringIO(sbjct_record), "fasta")

                with open(f"./proteins/{prot}/sequences.txt", "a") as f:
                    f.write(str(seq_record.seq)+"\n")


########################################################################################################

def tokenize_the_sequence(seq):
    """ tokenize the sequence 

    @param seq: the sequence to tokenize
    @type seq: str

    @return: the tokenized sequence
    @rtype: tensor
    
    """
    # start counting from 0
    # 20 == 21
    number_of_amino_acids = 20

    return torch.tensor([hash(char) % number_of_amino_acids for char in str(seq.seq)], dtype=torch.int8)

########################################################################################################

def tokenize_per_protein(protein):
    """ tokenize per protein 

    @param protein: the protein to tokenize
    @type protein: str

    @return: the tokenized protein
    @rtype: list
    
    """
    return [tokenize_the_sequence(SeqIO.read(file, "fasta")) for file in protein]

########################################################################################################

def tokenize_all_proteins(proteins):
    """ tokenize all proteins 

    @param proteins: the proteins to tokenize
    @type proteins: list

    @return: the tokenized proteins
    @rtype: list
    
    """
    return [tokenize_per_protein(protein) for protein in proteins]


########################################################################################################

# padd all the squences to the same length
def padd_the_sequences(sequences, max_len):
    """ padd the sequences

    @param sequences: the sequences to padd
    @type sequences: list

    @param max_len: the max length of the sequences
    @type max_len: int

    @return: the padded sequences
    @rtype: list
    
    """

    return [torch.nn.functional.pad(seq+1, (0, max_len - len(seq)), 'constant', 0) for seq in sequences]

########################################################################################################

def from_xml_to_BPE_protein_tokenization():
    """Encode the sequences using BPE.

    Args:
        None

    Returns:    
        None
    
    """

    # Open the BLAST XML output file
    blast_xml_files = glob.glob("./proteins/*/*-blastp.xml")

    # vocab size
    # vocab_s = dict([(file.split('/')[2],max([len(seq) for seq in open(file, 'r').readlines()])*15) for file in glob.glob("./proteins/*/sequences/sequences.txt")])
    vocab_s = dict([(file.split('/')[2],max([len(seq) for seq in open(file, 'r').readlines()])*15) for file in glob.glob("./proteins/*/sequences/sequences.txt")])

    vocab_s['insulin'] = 6000
    vocab_s['hemoglobin'] = 6722
    vocab_s['erythropoietin'] = 8000
    vocab_s['collagen'] = int(vocab_s['collagen']/3)
    vocab_s['myosin'] = int(vocab_s['myosin']/3)
    vocab_s['trypsin'] = int(vocab_s['trypsin']/2)
    vocab_s['elastin'] = 8000
    vocab_s['tubulin'] = 3421

    for blast_xml_file in blast_xml_files:

        # Extract the protein name from the file path
        prot = blast_xml_file.split("/")[2]

        # https://github.com/google/sentencepiece/blob/master/doc/options.md
        spm.SentencePieceTrainer.train(
            input=f'./proteins/{prot}/sequences/sequences.txt', 
            model_type='bpe', 
            shuffle_input_sentence=False,
            split_by_whitespace=False,
            max_sentencepiece_length=16,
            allow_whitespace_only_pieces=True,
            model_prefix=f'BPE_model_{prot}', 
            vocab_size=vocab_s[prot]
        )

        sm = spm.SentencePieceProcessor()
        # load the model
        sm.load(f'BPE_model_{prot}.model')

        # Parse the BLAST output file
        with open(blast_xml_file) as blast_xml:
            blast_record = NCBIXML.read(blast_xml)

        # Extract the sequences from the BLAST output file and write to FASTA file
        sequences = []
        sequences_str = []

        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:

                # Format the sbjct sequence as a FASTA record
                accession = alignment.hit_id.split("|")[1]
                sbjct_record = ">{} {}_{}\n{}\n".format(accession,alignment.hit_def, hsp.sbjct_start, hsp.sbjct)

                # Parse the FASTA record using SeqIO.read()
                seq_record = SeqIO.read(StringIO(sbjct_record), "fasta")
                
                # encode the sequence
                sequences_str.append((accession,sm.encode(str(seq_record.seq), out_type=str)))
                sequences.append((accession,sm.encode(str(seq_record.seq))))

        # Write the encoded sequences into a pickle file
        with open(f"./proteins/{prot}/sequences/sequences_BPE.pkl", "wb") as f:
            pickle.dump(sequences, f)
        with open(f"./proteins/{prot}/sequences/sequences_BPE_str.pkl", "wb") as f:
            pickle.dump(sequences_str, f)

        # move the model+vocab to the right folder
        os.rename(f"./BPE_model_{prot}.model", f"./proteins/{prot}/sequences/BPE_model_{prot}.model")
        os.rename(f"./BPE_model_{prot}.vocab", f"./proteins/{prot}/sequences/BPE_model_{prot}.vocab")

########################################################################################################

def get_batch(sequences_token, idx):
    """ Function to get the batch of sequences

    Returns:
        list: list with all the sequences
    """
    if idx == 5:
        yield [sequences_token[m:M]
                    for m, M in zip(np.linspace(0, 488, 60)[:59].astype(int),np.linspace(0, 488, 60)[1:].astype(int))]
        
    elif idx == 6:
        yield [sequences_token[m:M]
                    for m, M in zip(np.linspace(0, 498, 60)[:59].astype(int),np.linspace(0, 498, 60)[1:].astype(int))]
    elif idx == 7:
        yield [sequences_token[m:M]
                    for m, M in zip(np.linspace(0, 489, 60)[:59].astype(int),np.linspace(0, 489, 60)[1:].astype(int))]
        
    else:
        yield [sequences_token[m:M]
                    for m, M in zip(np.linspace(0, 500, 80)[:79].astype(int),np.linspace(0, 500, 80)[1:].astype(int))]
        
########################################################################################################

def get_esm_encoding(batch_tokens, model, device):
    # allocate the token on the GPU
    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    yield results["representations"][33].type(torch.float16).cpu()

########################################################################################################

def extract_embedding(sequences_token, model, idx):
    """ Function to extract the embedding from the sequences

    Args:
        sequences_token (list): list with all the sequences
        model (torch.nn.Module): model to use to extract the embedding
        alphabet (esm.Alphabet): alphabet of the model
        batch_converter (esm.pretrained.esm1_t6_43M_UR50S): converter of the model

    Returns:
        list: list with all the embedding
    """

    # list with all the extracted representations
    token_representations = []

    for batch_tokens in list(get_batch(sequences_token, idx))[0]:

        # extract the embedding
        token_representations.append(list(get_esm_encoding(batch_tokens, model, device))[0])

        # free the memory
        gc.collect()
        del batch_tokens

    return token_representations

########################################################################################################

def prt_assc_counts(ns2assc):
    """Print the number of genes and GO IDs in an association"""
    for nspc, gene2goids in sorted(ns2assc.items()):
        print("{NS} {N:6,} genes, {GOs:6,} GOs".format(
            NS=nspc, N=len(gene2goids), GOs=len(set.union(*gene2goids.values()))))
        
########################################################################################################

number_of_amino_acids = 12533
def get_dict_all_go_tensor(ns2assc_all):
    """This function returns a dictionary with all the go terms encoded
    in a tensor with a vocab of 12533 (number of amino acids).
    
    """
    dictionary_all = {}
    for fun in ns2assc_all.keys():
        dictionary = {}
        for i in ns2assc_all[fun].keys():
            temp = []
            try:
                for j in list(ns2assc_all['MF'][i]):
                    temp.append(hash(j) % number_of_amino_acids)
                dictionary[i] = torch.from_numpy(np.array(temp)).type(torch.int16)
            except:
                pass
        dictionary_all[fun] = dictionary
    return dictionary_all

########################################################################################################

def get_dict_all_go_tensor_padded(ns2assc_all):
    """This function returns a dictionary with all the go terms encoded
    in a tensor with a vocab of 12533 (number of amino acids) padded.
    
    """
    dictionary_all = {}
    for fun in ns2assc_all.keys():
        dictionary = {}
        for i in ns2assc_all[fun].keys():
            temp = []
            try:
                for j in list(ns2assc_all['MF'][i]):
                    temp.append(hash(j) % number_of_amino_acids)
                dictionary[i] = torch.nn.functional.pad(
                                            torch.from_numpy(
                                                    np.array(sorted(temp))
                                                ).type(torch.int16), (0, max_lenght - len(temp)))
            except:
                pass
        dictionary_all[fun] = dictionary
    return dictionary_all


########################################################################################################

class GoTermDataset(torch.utils.data.Dataset):
    def __init__(self, go_term, go_term_type):
        self.go_term = go_term[go_term_type]
        self.go_term_type = go_term_type
        self.len = len(go_term[go_term_type])

    def __len__(self):
        return len(self.go_term)
    
    def __getnumberofitems__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        go_term = self.go_term[idx]
        go_term_type = self.go_term_type[idx]

        return go_term, go_term_type
    
########################################################################################################

def retrieve_sequences(geneID_list):
    """ Retrieve the sequences of the genes and the accession number.
    Then store them in a dictionary.
    
    Args:
        geneID_list (np.array): list of geneID to retrieve the sequences.
        
    Returns:
        dict_sequences (dict): dictionary with the geneID as key and the
            sequence and the accession number as values.
    
    
    """
    
    dict_sequences = {}
    for geneID in tqdm(geneID_list):
        
        # temporary dictionary
        temp = {}
        
        # get the accession number
        handle = Entrez.esearch(db="nucleotide", term=f"gene_id:{geneID}")
        record = Entrez.read(handle)
        accession = record["IdList"][0]
        
        # fetch the sequence
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        
        # store the informtions in a dictionary
        temp['accession'] = accession
        temp['sequence'] = str(record.seq)
        
        dict_sequences[geneID] = temp
        
        with open('./proteins/go_term/dict_retrieved_genes_checkpoint56.pkl', 'wb') as f:
            pickle.dump(dict_sequences, f)
        
    return dict_sequences

########################################################################################################

def retrieve_sequences_1(geneID):
    """ Retrieve the sequences of the genes and the accession number.
    Then store them in a dictionary.
    
    Args:
        geneID_list (np.array): list of geneID to retrieve the sequences.
        
    Returns:
        dict_sequences (dict): dictionary with the geneID as key and the
            sequence and the accession number as values.
    
    
    """
    # temporary dictionary
    temp = {}
    
    # get the accession number
    handle = Entrez.esearch(db="nucleotide", term=f"gene_id:{geneID}")
    record = Entrez.read(handle)
    accession = record["IdList"][0]
    
    # fetch the sequence
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    
    # store the informtions in a dictionary
    temp['accession'] = accession
    temp['sequence'] = str(record.seq)
                
    return (geneID, temp)

########################################################################################################

class BertLike_module(pl.LightningModule):
    def __init__(self, num_tokens, hidden_size, num_layers, num_heads, dropout_rate):
        super(BertLike, self).__init__()
        
        # tokenizer
        self.token_embedding = nn.Embedding(num_tokens, hidden_size)
        
        # positional embedding
        self.position_embedding = nn.Embedding(1000, hidden_size)
        
        # transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=4*hidden_size, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # get the out tokens
        self.fc = nn.Linear(hidden_size, num_tokens)

    def forward(self, x):
        
        # token_emb (batch_size, seq_len, hidden_size)
        # get the token embedding
        token_emb = self.token_embedding(x) 
        
        # position_emb (1, seq_len, hidden_size)
        position_emb = self.position_embedding(torch.arange(x.size(1), device=x.device))[None, :, :] 
        
        # embedding (batch_size, seq_len, hidden_size)
        # add position embedding to token embedding
        emb = self.dropout(token_emb + position_emb) 
        
        # bert like block
        for layer in self.encoder_layers:
            emb = layer(emb)
            
        # logits (batch_size, num_tokens)
        # get the out tokens
        logits = self.fc(emb[:, -1, :]) 
        
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get the validation data
        x, y = batch
        
        # get the predictions
        logits = self(x)
        
        # loss
        loss = F.cross_entropy(logits, y)
        
        # log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        
        # log metrics
        self.log('val_acc', acc, on_step=True, on_epoch=True)
        
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
