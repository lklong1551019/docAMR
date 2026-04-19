import penman
import json
import os
from tqdm import tqdm

def get_official_amr_relations(max_snt=0):
    """Returns a base set of standard AMR 1.2 relations to guarantee coverage."""
    core_roles = [f":ARG{i}" for i in range(10)]
    
    # Official non-core relations from AMR guidelines
    non_core = [
        ":accompanier", ":age", ":beneficiary", ":cause", ":concession", 
        ":condition", ":consist-of", ":degree", ":destination", ":direction", 
        ":domain", ":duration", ":example", ":extent", ":frequency", 
        ":instrument", ":li", ":location", ":manner", ":medium", ":mod", 
        ":mode", ":name", ":part", ":path", ":polarity", ":politeness", 
        ":poss", ":purpose", ":quant", ":scale", ":source", ":subevent", 
        ":time", ":topic", ":value", ":ord", ":weekday", ":dayperiod", 
        ":month", ":day", ":year", ":timezone", ":quarter", ":decade", ":era"
    ]
    
    # AMR allows adding "-of" to almost any relation to invert it
    inverse_roles = [f"{role}-of" for role in core_roles + non_core]
    
    # Common op roles for lists/names
    op_roles = [f":op{i}" for i in range(1, 101)]

    # DocAMR specific tokens
    docamr_roles = [":same-as"]
    if max_snt > 0:
        docamr_roles += [f":snt{i}" for i in range(1, max_snt + 1)]
    
    return set(core_roles + non_core + inverse_roles + op_roles + docamr_roles)

def extract_dataset_relations(file_paths):
    """Parses your actual docAMR files to find dataset-specific relations and max sentence count."""
    dataset_relations = set()
    max_snt = 0
    
    print(f"Scanning {len(file_paths)} files for relations...")
    for path in tqdm(file_paths, desc="Processing files"):
        try:
            # Load graphs using penman
            with open(path, "r", encoding="utf-8") as f:
                # Some files might have multiple graphs separated by newlines
                # penman.load reads the first one, penman.iter reads all
                graphs = penman.iter(f)
                
                for graph in graphs:
                    for edge in graph.edges():
                        # edge.role contains the relation (e.g., ":ARG0")
                        role = edge.role
                        dataset_relations.add(role)
                        
                        # Check for :sntN markers to find max sentence index
                        if role.startswith(":snt"):
                            try:
                                snt_num = int(role[4:])
                                if snt_num > max_snt:
                                    max_snt = snt_num
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Warning: Could not read {path}. Error: {e}")
            
    return dataset_relations, max_snt

def main():
    # 1. Automatically find all .out files in the output_doc_amr subdirectories
    input_dir = "output_doc_amr"
    dataset_files = []
    
    if not os.path.exists(input_dir):
        # Try finding it relative to the script location if not in CWD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir, "output_doc_amr")
    
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".out") and not file.endswith(".coref"):
                    dataset_files.append(os.path.join(root, file))
    
    if not dataset_files:
        print(f"No .out files found in {input_dir}. Current working directory: {os.getcwd()}")
        return

    print(f"Found {len(dataset_files)} AMR output files in subdirectories of {input_dir}")
    
    # 2. Get both sets of relations
    dataset_roles, max_snt = extract_dataset_relations(dataset_files)
    official_roles = get_official_amr_relations(max_snt)
    
    print(f"\nDetected maximum sentence index in dataset: {max_snt}")
    print(f"Found {len(official_roles)} official and synthetic AMR relations (including :snt1 to :snt{max_snt}).")
    print(f"Found {len(dataset_roles)} unique relations actually used in your datasets.")
    
    # 3. COMBINE THEM (Union of both sets)
    combined_roles = official_roles.union(dataset_roles)
    
    # Sort them alphabetically for clean output
    final_vocab_list = sorted(list(combined_roles))
    
    print(f"\nTotal unique relations after combining: {len(final_vocab_list)}")
    
    # 4. Save to a JSON file so your DiffuSeq repo can easily load it
    output_file = "custom_amr_relations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_vocab_list, f, indent=4)
        
    print(f"\nSuccessfully saved the combined vocabulary to {output_file}!")

if __name__ == "__main__":
    main()
