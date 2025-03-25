import pandas as pd
import os
from mistralai import Mistral
from dotenv import load_dotenv
from docx import Document

load_dotenv()

# Load model API
api_key = os.environ.get('api_key')
llm_model = os.environ.get('llm_model')
embeddings_model = os.environ.get("embeddings_model")

client = Mistral(api_key=api_key)


# Load all sheets into a dictionary of DataFrames
all_sheets = pd.read_excel(r"./data/Incident_Data.xlsx", sheet_name=None)
df_incidents_embed = pd.read_pickle(r"./data/incidence_desc_emb_df.pkl")

df_incidents, df_configuration_data, df_kb_articles, df_recommendations = None, None, None, None

# Create variables dynamically for each sheet
for sheet_name, df in all_sheets.items():
    var_name = f"df_{sheet_name.lower()}"
    globals()[var_name] = df
    print(f"Created DataFrame: {var_name}")


# Load SOPs
def read_word_files(directory):
    sop_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith(".docx"):  # Only process .docx files
            file_path = os.path.join(directory, filename)
            try:
                doc = Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
                sop_dict[filename] = content
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return sop_dict


# Example usage
sop_path = r"./data/IT_Support_SOPs"
sop_dict = read_word_files(sop_path)

# Path for Ansible Script
ansible_path = r"./data/Ansible_Scripts_For_Metrics"