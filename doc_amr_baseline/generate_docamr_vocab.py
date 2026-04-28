import penman
import json
import os
import re
from tqdm import tqdm

def load_bert_vocab(filepath):
    """Loads the BERT vocabulary into a set for fast O(1) lookup."""
    vocab = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                vocab.add(line.strip())
        print(f"Loaded {len(vocab)} standard tokens from {os.path.basename(filepath)}.")
    else:
        print(f"[WARN] BERT vocab not found at {filepath}. Proceeding without standard token filtering.")
    return vocab

def extract_dataset_tokens(file_paths):
    """Parses DocAMR files using penman to find relations and high-value concepts."""
    dataset_tokens = set()
    dataset_tokens_simple = set()
    
    # Core structural nodes essential for the document graph
    core_structures = {
        'person', 'thing', 'date-entity', 'government-organization', 
        'amr-unknown', 'document'
    }
    
    print(f"Scanning {len(file_paths)} files for relations and concepts...")
    for path in tqdm(file_paths, desc="Processing files"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # --- SANITIZATION FIX ---
            sanitized_content = re.sub(r'/\s*(\d+/\d+)', r'/ "\1"', content)
            
            try:
                graphs = penman.loads(sanitized_content)
            except Exception as e:
                print(f"Warning: Could not parse sanitized content of {path}. Error: {e}")
                continue
                
            for graph in graphs:
                # 1. Extract Relations (Edges)
                for edge in graph.edges():
                    if edge.role == ':instance':
                        continue 
                        
                    role = edge.role
                    dataset_tokens.add(role)
                    
                    # SIMPLE RULE: Strip all digits from relations
                    role_simple = re.sub(r'\d+', '', role)
                    dataset_tokens_simple.add(role_simple)
                    
                # 2. Extract Concepts (Instances)
                for instance in graph.instances():
                    concept = str(instance.target)
                    
                    # STRICT FILTER: Only keep frames (-01) or core structures
                    if re.search(r'-[0-9]{2,}$', concept) or concept in core_structures:
                        dataset_tokens.add(concept)
                        
                        # SIMPLE RULE: Strip the frame number suffix
                        concept_simple = re.sub(r'-\d+$', '', concept)
                        dataset_tokens_simple.add(concept_simple) 

        except Exception as e:
            print(f"Warning: Could not read {path}. Error: {e}")
            
    return dataset_tokens, dataset_tokens_simple

def load_custom_vocab(filepath):
    """Safely loads existing JSON vocabularies."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = set(json.load(f))
            print(f"Loaded {len(loaded)} tokens from {os.path.basename(filepath)}")
            return loaded
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    else:
        print(f"File not found: {filepath}. Starting with empty set.")
    return set()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 0. Load the standard BERT Vocabulary
    bert_vocab_path = os.path.join(script_dir, "multi-bert-base-cased-vocab.txt")
    bert_vocab = load_bert_vocab(bert_vocab_path)
    
    # 1. Automatically find all .out and .amr files in output_doc_amr
    input_dir = os.path.join(script_dir, "output_doc_amr")
    dataset_files = []
    
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if (file.endswith(".out") or file.endswith(".amr")) and not file.endswith(".coref"):
                    dataset_files.append(os.path.join(root, file))
    
    if not dataset_files:
        print(f"No valid AMR files found in {input_dir}.")
        return

    print(f"Found {len(dataset_files)} AMR output files.")
    
    # 2. Extract tokens from DocAMR graphs
    docamr_tokens, docamr_tokens_simple = extract_dataset_tokens(dataset_files)
    print(f"\nExtracted {len(docamr_tokens)} unique structure/frame tokens from DocAMRs.")
    
    # 3. Load previous custom tokens from output_dataset_amr
    base_vocab_dir = os.path.join(script_dir, "output_dataset_amr")
    amrs_file = os.path.join(base_vocab_dir, "amrs_token.json")
    amrs_simple_file = os.path.join(base_vocab_dir, "amrs_token_simple.json")
    
    base_tokens = load_custom_vocab(amrs_file)
    base_tokens_simple = load_custom_vocab(amrs_simple_file)
    
    # 4. UNION Both Sets
    combined_tokens = docamr_tokens.union(base_tokens)
    combined_tokens_simple = docamr_tokens_simple.union(base_tokens_simple)
    
    # 5. THE BERT FILTER: Remove any token that already exists in BERT
    final_vocab = sorted([t for t in combined_tokens if t not in bert_vocab])
    final_vocab_simple = sorted([t for t in combined_tokens_simple if t not in bert_vocab])
    
    print(f"\nFinal unified vocabulary size (Full): {len(final_vocab)}")
    print(f"Final simplified vocabulary size (Simple): {len(final_vocab_simple)}")
    
    # 6. Save output files directly into output_doc_amr
    out_file = os.path.join(input_dir, "doc_amrs_token.json")
    out_simple_file = os.path.join(input_dir, "doc_amrs_token_simple.json")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, indent=4)
        
    with open(out_simple_file, "w", encoding="utf-8") as f:
        json.dump(final_vocab_simple, f, indent=4)
        
    print(f"\nSuccessfully saved ultra-clean vocabularies to '{out_file}' and '{out_simple_file}'!")

if __name__ == "__main__":
    main()