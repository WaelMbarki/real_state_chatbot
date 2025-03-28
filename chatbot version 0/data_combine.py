import json
def concat_json_files(file1, file2, file3, output_file):
    data = []
    
    for file in [file1, file2, file3]:
        with open(file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))  # Assuming JSON files contain lists
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Example usage
concat_json_files('real_state_en.json', 'registration_en.json', 'laws_en.json', 'data_en.json')
concat_json_files('real_state_fr.json', 'registration_fr.json', 'laws_fr.json', 'data_fr.json')
concat_json_files('real_state_ar.json', 'registration_ar.json', 'laws_ar.json', 'data_ar.json')
