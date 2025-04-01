import json
import os
import numpy as np
import openmm as mm

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, 'data_files')

def convert_keys_to_int(data):
    """
    Recursively converts the keys of a dictionary to integers if possible.
    Args:
        data (dict or list): The data structure whose keys need to be converted.
    
    Returns:
        Converted dictionary or list.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            try:
                new_key = int(key)
            except (ValueError, TypeError):
                new_key = key
            new_dict[new_key] = convert_keys_to_int(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_keys_to_int(item) for item in data]
    else:
        return data

def load_json_data(file_name):
    """
    Loads and parses a JSON file, converting string keys to integers where appropriate.
    
    Args:
        file_name (str): The name of the JSON file to load.
        
    Returns:
        Parsed data from the JSON file with keys converted to integers.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found in {DATA_DIR}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {file_name}: {e}")
    
    return convert_keys_to_int(data)

# Load Nucleosome and DNA data from JSON files
NUCLEOSOME_DATA = load_json_data('nucleosome_data.json')
TETRAMER_DATA = load_json_data('DNA_bonds_angles.json')

# Platform and precision configuration
try:
    PLATFORM = mm.Platform.getPlatformByName('CUDA')
    PROPERTIES = {'Precision': 'mixed'}
 
except Exception as e:
    PLATFORM = None
    PROPERTIES = None
