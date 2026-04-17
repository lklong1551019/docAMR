#conda activate allen_nlp
import allennlp
from allennlp.predictors.predictor import Predictor
from itertools import accumulate
# import allennlp_models.tagging
import glob
import pickle
from tqdm import tqdm
import argparse
import os

# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
local_spanbert_path = "models/"

# Use overrides to bypass the broken network calls

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz", overrides={

    # 1. Primary Dataset Reader
    "dataset_reader.token_indexers.tokens.model_name": local_spanbert_path,

    # 2. Validation Dataset Reader
    "validation_dataset_reader.token_indexers.tokens.model_name": local_spanbert_path,

    # 3. The Model Embedder itself
    "model.text_field_embedder.token_embedders.tokens.model_name": local_spanbert_path
})

def get_allen_coref(filepath,from_amr=False):
    """
    Extracts coreference chains from a document using the pre-loaded AllenNLP SpanBERT Large model.

    This function reads a document file, tokenizes it into sentences, and uses the AllenNLP 
    predictor to find coreference clusters (mentions referring to the same entity). It translates 
    the model's global token indices into a nested structure corresponding to sentence index 
    and the relative token span within that sentence.

    Args:
        filepath (str): Path to the input file containing the document text or formatted AMR parses.
        from_amr (bool, optional): If True, extracts sentences specifically from the `::tok` 
                                   metadata fields inside AMR format files. Defaults to False.

    Returns:
        list[list[list[int]]]: A list of coreference clusters. Each cluster is a list of entity mentions,
                               where each mention is defined as `[sentence_index, start_token_idx, end_token_idx]`.
    """
    
    f1 = open(filepath,'r').read()
    sen_list = f1.splitlines()
    if from_amr:
        sen_tok_list = [s.split('::tok ')[-1].split() for s in sen_list if '::tok' in s]
    else:
        sen_tok_list = [s.split() for s in sen_list]
    sen_tok_len = [len(l) for l in sen_tok_list]
    sen_tok_len = list(accumulate(sen_tok_len))
    sen_tok = [item for sublist in sen_tok_list for item in sublist]


    pred = predictor.predict_tokenized(tokenized_document=sen_tok)
    clusters = pred['clusters']
    document = pred['document']
    new_cluster = []

    for c in clusters:
        new_c = []
        for m in c:
            for idx,l in enumerate(sen_tok_len):
                if m[0]< l:
                    prev_len = sen_tok_len[idx-1]
                    break
            if idx!=0:
                new_c.append([idx,m[0]-prev_len,m[1]-prev_len])
            else:
                new_c.append([idx,m[0],m[1]])
        new_cluster.append(new_c)
    return new_cluster        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_sen',type=str)
    parser.add_argument('--path_to_out',type=str,help='path to output',default=None)
    parser.add_argument('--from_amr',action='store_true')
    parser.add_argument('--from_json',action='store_true')

    args = parser.parse_args()
    args.path_to_sen+='/'
   
    
    if args.from_amr:
        if len(glob.glob(args.path_to_sen+'*.amr')) > 0:
            ext = '.amr'
        elif len(glob.glob(args.path_to_sen+'*.parse')) > 0:
            ext = '.parse'
        else:
            ext = '.txt'
    else:
        ext = '.txt'
    
    path_fill = args.path_to_sen+'*'+ext

    for filepath in tqdm(glob.iglob(path_fill)):
        # Extract the document name (e.g., 'doc-1') from the full path. 
        # By separating by '/', this perfectly isolates files within varying parent folders (like 'dev2010.en-vi.en' vs 'train_en-vi.en').
        doc_id = filepath.split('/')[-1].split('.')[0]
        
        # Save parsed clusters independently per document inside the parent dataset folder.
        # This "streaming" logic replaces the approach of saving all files into a single huge dictionary, eliminating out-of-memory errors.
        if args.path_to_out is None:
            out_filepath = args.path_to_sen + doc_id + '.coref'
        else:
            out_filepath = args.path_to_out + '/' + doc_id + '.coref'
            
        # Check and skip execution if this specific document has already been processed and cached previously.
        if os.path.exists(out_filepath):
            continue
            
        # Log the file we are currently attempting to process. If it OOMs here, this log will reveal the culprit.
        status_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_processing_status.log")
        with open(status_log_path, "w") as status_log:
            status_log.write(f"Currently processing: {doc_id}\n")
            
        try:
            clusters = get_allen_coref(filepath,from_amr=args.from_amr)
            with open(out_filepath,'wb') as f2:
                pickle.dump(clusters,f2)
        except Exception as e:
            error_msg = f"Error processing document {doc_id} ({filepath}): {str(e)}\n"
            print(error_msg)
            
            # Log the error to a file
            log_file = "coref_errors.log"
            if args.path_to_out:
                log_file = os.path.join(args.path_to_out, log_file)
            with open(log_file, "a") as err_log:
                import traceback
                err_log.write(error_msg)
                err_log.write(traceback.format_exc() + "\n")

