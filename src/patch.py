import json

def patch_notebook():
    with open('Copy of MentalHealthEmotionDetection.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Modify dataset loading
            if 'dataset = load_dataset("go_emotions")' in source and 'REDUCED DATASET' not in source:
                new_source = source + '\n\n# --- FAST MODE ---'
                new_source += '\nfor split in dataset.keys():'
                new_source += '\n    dataset[split] = dataset[split].select(range(min(150, len(dataset[split]))))'
                new_source += '\nprint("REDUCED DATASET FOR FAST EXECUTION:", dataset)\n'
                # Reconstruct source array
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                
            # Modify training arguments
            if 'TrainingArguments(' in source and 'num_train_epochs=3' in source:
                new_source = source.replace('num_train_epochs=3', 'num_train_epochs=1')
                cell['source'] = [line + '\n' for line in new_source.split('\n')]

    with open('Copy of MentalHealthEmotionDetection.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print('Notebook patched successfully.')

if __name__ == '__main__':
    patch_notebook()
