import penman
import json
import os
import re
from tqdm import tqdm

def extract_dataset_relations(file_paths):
    """Parses your actual docAMR files to find dataset-specific relations."""
    dataset_relations = set()
    dataset_relations_simple = set()
    
    print(f"Scanning {len(file_paths)} files for relations...")
    for path in tqdm(file_paths, desc="Processing files"):
        try:
            # Load graphs using penman
            with open(path, "r", encoding="utf-8") as f:
                # Some files might have multiple graphs separated by newlines
                try:
                    graphs = penman.iterdecode(f)
                except AttributeError:
                    # Fallback if somehow it's not what we expect
                    graphs = [penman.load(f)]
                
                for graph in graphs:
                    for edge in graph.edges():
                        role = edge.role
                        dataset_relations.add(role)
                        
                        role_simple = re.sub(r'\d+', '', role)
                        dataset_relations_simple.add(role_simple)
        except Exception as e:
            print(f"Warning: Could not read {path}. Error: {e}")
            
    return dataset_relations, dataset_relations_simple

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
    dataset_roles, dataset_roles_simple = extract_dataset_relations(dataset_files)
    
    print(f"\nFound {len(dataset_roles)} unique relations actually used in your datasets.")
    
    # 2.5 Load custom relations from output_dataset_amr
    amrs_relations_file = "output_dataset_amr/custom_relation_amrs.json"
    amrs_relations_simple_file = "output_dataset_amr/custom_relation_amrs_simple.json"
    
    amrs_roles = set()
    amrs_roles_simple = set()
    
    if not os.path.exists("output_dataset_amr"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        amrs_relations_file = os.path.join(script_dir, "output_dataset_amr", "custom_relation_amrs.json")
        amrs_relations_simple_file = os.path.join(script_dir, "output_dataset_amr", "custom_relation_amrs_simple.json")
        
    try:
        if os.path.exists(amrs_relations_file):
            with open(amrs_relations_file, "r", encoding="utf-8") as f:
                loaded_roles = json.load(f)
                amrs_roles.update(loaded_roles)
            print(f"Loaded {len(loaded_roles)} relations from {amrs_relations_file}")
    except Exception as e:
        print(f"Warning: Could not load {amrs_relations_file}: {e}")
        
    try:
        if os.path.exists(amrs_relations_simple_file):
            with open(amrs_relations_simple_file, "r", encoding="utf-8") as f:
                loaded_roles_simple = json.load(f)
                amrs_roles_simple.update(loaded_roles_simple)
            print(f"Loaded {len(loaded_roles_simple)} relations from {amrs_relations_simple_file}")
    except Exception as e:
        print(f"Warning: Could not load {amrs_relations_simple_file}: {e}")

    # Ensure simple logic is fully applied to amrs_roles as well
    for role in amrs_roles:
        amrs_roles_simple.add(re.sub(r'\d+', '', role))
    
    # 3. COMBINE THEM (Union of both sets)
    combined_roles = dataset_roles.union(amrs_roles)
    combined_roles_simple = dataset_roles_simple.union(amrs_roles_simple)
    
    # Sort them alphabetically for clean output
    final_vocab_list = sorted(list(combined_roles))
    final_vocab_list_simple = sorted(list(combined_roles_simple))
    
    print(f"\nTotal unique relations after combining: {len(final_vocab_list)}")
    print(f"Total simple unique relations after combining: {len(final_vocab_list_simple)}")
    
    # 4. Save to JSON files
    output_file_full = "custom_relation_docamrs.json"
    with open(output_file_full, "w", encoding="utf-8") as f:
        json.dump(final_vocab_list, f, indent=4)
        
    output_file_simple = "custom_relation_docamrs_simple.json"
    with open(output_file_simple, "w", encoding="utf-8") as f:
        json.dump(final_vocab_list_simple, f, indent=4)
        
    print(f"\nSuccessfully saved the combined vocabulary to {output_file_full} and {output_file_simple}!")

if __name__ == "__main__":
    main()
