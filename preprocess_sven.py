import json
import glob
import os
import argparse
from tqdm import tqdm

def process_sven_files(input_dir, output_file):
    """
    Process multiple SVEN data files into a single JSONL file
    """
    # Find all JSON or JSONL files in the input directory
    file_paths = glob.glob(os.path.join(input_dir, "*.json*"))
    
    print(f"Found {len(file_paths)} files to process")
    
    # Open output file
    with open(output_file, 'w') as outfile:
        count = 0
        
        # Process each file
        for file_path in tqdm(file_paths):
            try:
                # Handle both single JSON objects and JSONL files
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as infile:
                        data = json.load(infile)
                        if isinstance(data, list):
                            for item in data:
                                outfile.write(json.dumps(item) + '\n')
                                count += 1
                        else:
                            outfile.write(json.dumps(data) + '\n')
                            count += 1
                else:  # JSONL file
                    with open(file_path, 'r') as infile:
                        for line in infile:
                            if line.strip():  # Skip empty lines
                                try:
                                    item = json.loads(line.strip())
                                    outfile.write(json.dumps(item) + '\n')
                                    count += 1
                                except json.JSONDecodeError:
                                    print(f"Error parsing line in {file_path}: {line}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    print(f"Processed {count} records and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SVEN data files into a single JSONL file')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing SVEN data files')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSONL file path')
    
    args = parser.parse_args()
    process_sven_files(args.input_dir, args.output_file)