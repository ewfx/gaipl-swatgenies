from typing import Any

import pandas as pd
import numpy as np
import openai
import os
from sklearn.metrics.pairwise import cosine_similarity
from registered_files import llm_model, embeddings_model, client, df_incidents, df_configuration_data, df_recommendations, df_kb_articles, df_incidents_embed

openai.api_key = os.getenv("OPENAI_API_KEY")
unique_incident_ids = df_incidents[df_incidents['status']=='Open']['incident_id'].unique().tolist()
unique_application_ids = df_incidents['application'].unique().tolist()


def llm_response(input_text, chatbot_activation=False):
    if chatbot_activation is True:
        chat_response = client.chat.complete(model=llm_model, messages=input_text)
        return chat_response.choices[0].message.content
    else:
        chat_response = client.chat.complete(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": input_text,
                },
            ]
        )
        return chat_response.choices[0].message.content

def create_embeddings(texts):
    """
    Generates embeddings for the given text(s) using the OpenAI API.
    Args:
        texts: A string or a list of strings to generate embeddings for.
    Returns:
        A list of embeddings, or an empty list if an error occurred.
    """
    if not openai.api_key:
        print("Error: OpenAI API key not found in environment variables.")
        return []

    if isinstance(texts, str):
        texts = [texts]  # make into a list if it is a single string.
    elif not isinstance(texts, list):
        print("Error: Input must be a string or a list of strings.")
        return []

    try:
        response = openai.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        # Correct way to access embeddings:
        embeddings = [item.embedding for item in response.data]

        # Access using object attributes.
        return embeddings

    except openai.RateLimitError as e:
        print(f"OpenAI Rate limit error: {e}")
    # Handle rate limit error (e.g., retry after a delay)
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication error: {e}")
        # handle authentication errors.
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        # Handle other OpenAI API errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def save_incidence_embeddings():
    """
    Generates and saves embeddings for incident descriptions.

    This function takes the 'description' column from the 'df_incidents' DataFrame,
    creates embeddings for each description using the 'create_embeddings' function,
    and saves the resulting embeddings to a pickle file.
    """

    # Generate embeddings for the incident descriptions.
    embeddings_incidence = create_embeddings(df_incidents['description'].tolist())

    # Create a Pandas DataFrame from the generated embeddings, using 'incident_id' as the index.
    incidence_desc_emb_df = pd.DataFrame(embeddings_incidence, index=df_incidents['incident_id'])

    # Save the DataFrame containing incident description embeddings to a pickle file.
    # The file path is specified as './data/incidence_desc_emb_df_new.pkl'.
    incidence_desc_emb_df.to_pickle(r'./data/incidence_desc_emb_df_new.pkl')

    print("Embeddings Saved")


def get_nearest_match(user_input, top_n=50):
    """
    Finds the nearest matching incidents to the user's input based on cosine similarity.

    Args:
        user_input (str): The user's input string.
        top_n (int, optional): The number of top matches to consider initially. Defaults to 50.

    Returns:
        list: A list of indices of the nearest matching incidents.
    """
    # Preprocess the user input: lowercase and remove leading/trailing spaces.
    user_input = [user_input.strip().lower()]

    # Generate embeddings for the user input.
    user_emb = create_embeddings(user_input)

    # Calculate cosine similarity between user input embedding and all incident embeddings.
    sim = cosine_similarity(np.array(user_emb[0]).reshape(1, -1), np.array(df_incidents_embed.values)).flatten()

    # Get the indices of the top_n most similar incidents.
    top_n_indices = np.argsort(sim)[::-1][:top_n]

    # Create a DataFrame containing the top_n incidents and their similarity scores.
    top_n_df = df_incidents_embed.iloc[top_n_indices]
    top_n_df.loc[:,'similarity_score'] = sim[top_n_indices]

    # Initialize the starting similarity score for filtering.
    starting_sim_score = 1

    # Iteratively reduce the similarity score threshold until a reasonable number of matches are found.
    while True:
        # Reduce the similarity score threshold by 0.1 in each iteration.
        starting_sim_score -= 0.1

        # Filter the top_n DataFrame to keep only incidents with similarity scores above the current threshold.
        top_n_df_temp = top_n_df[top_n_df['similarity_score']>starting_sim_score]

        # Break the loop if matches are found or the threshold reaches 0.6.
        if len(top_n_df_temp) > 0 or starting_sim_score == 0.6:
            break

    return top_n_df_temp.index.tolist()


def check_duplicate_incidents(df_main):
    """
    Checks for potential duplicate incidents in the DataFrame.
    Duplicate incidents are identified based on matching 'application' and 'created_at' values,
    specifically within 'Open' status incidents.
    Args:
        df_main (pd.DataFrame): The DataFrame containing incident data.
    Returns:
        pd.DataFrame: A modified DataFrame with a 'Duplicate Tickets' column indicating potential duplicates.
    """

    # Initialize an empty dictionary to store duplicate incident information.
    dup_dict = {}

    # Group 'Open' status incidents by 'application' and 'created_at', and aggregate 'incident_id' into sets.
    # This identifies groups of incidents with the same application and creation time.
    dup_list = df_main[df_main.status == 'Open'].groupby(['application', 'created_at']).agg({'incident_id': set})['incident_id'].tolist()

    # Iterate through each group of incident IDs.
    for item_ in dup_list:
        # Check if the group contains more than one incident ID, indicating a potential duplicate.
        if len(item_) > 1:
            # Iterate through each incident ID in the duplicate group.
            for id_ in item_:
                # Create a string describing the potential duplicate relationship.
                # The string indicates which other incident IDs are potential duplicates.
                dup_dict[id_] = f"{item_ - {id_} }"

    # Create a DataFrame from the 'dup_dict' dictionary.
    # This DataFrame maps each incident ID to its potential duplicate information.
    dup_df = pd.DataFrame(dup_dict.items(), columns=['incident_id', 'Suspected_Duplicate_Ticket'])

    # Merge the 'dup_df' DataFrame into the original 'df_main' DataFrame.
    # This adds a 'Duplicate Tickets' column to 'df_main', indicating potential duplicates for each incident.
    df_main_u = df_main.merge(dup_df, on='incident_id', how='left')

    return df_main_u


def affected_areas(df_incident_filtered: pd.DataFrame,
                          incident_id: str) -> None | tuple[Any, Any] | bool:
    """
    Analyzes incident metrics and compares them against configuration thresholds to determine affected areas.

    Args:
        df_incident_filtered: Pandas DataFrame containing filtered incident data.
        df_configuration_data: Pandas DataFrame containing configuration thresholds for CIs.
        incident_id: The incident ID to analyze (default: "INC-3001").

    Returns:
        bool: True if any metric exceeded its threshold, False otherwise.
    """

    df_metrics = df_incident_filtered[df_incident_filtered.incident_id == incident_id]

    # get Priority of the ticket
    affected_zones = df_metrics[['Priority']].to_dict(orient='records')[0]

    if 'CI_Affected' in df_metrics.columns:
        df_metrics = df_metrics.rename(columns={'CI_Affected': 'CI_Name'})

    metrics_list = ['CI_Name', 'cpu_usage', 'memory_usage', 'disk_usage',
                    'network_latency', 'Data Ingestion Delay', 'App Response Time',
                    'Query Latency', 'Auth Failures/sec', 'Blocked Connections',
                    'Queue Length', 'Battery Temp']

    # Check if all needed metrics exist in df_metrics
    if not all(col in df_metrics.columns for col in metrics_list):
        print(f"Warning: Missing metrics in incident data for {incident_id}")
        return False  # or raise an exception

    df_metrics = df_metrics[metrics_list].copy()  # using copy to avoid slice issue.
    df_metrics.set_index('CI_Name', inplace=True)

    if df_metrics.empty:
        print(f"Warning: No metrics found for incident {incident_id}")
        return False

    ci_name = df_metrics.index[0].strip()
    df_threshold = df_configuration_data[df_configuration_data.CI_Name.str.strip() == ci_name]

    # Check if all needed metrics exist in df_threshold
    if not all(col in df_threshold.columns for col in metrics_list):
        print(f"Warning: Missing metrics in configuration data for {ci_name}")
        return False

    df_threshold = df_threshold[metrics_list].copy()
    df_threshold.set_index('CI_Name', inplace=True)

    if df_threshold.empty:
        print(f"Warning: No thresholds found for CI {ci_name}")
        return False

    exceeds = (df_metrics > df_threshold)
    any_exceed = exceeds.any(axis=1).any()

    if any_exceed:
        active_metrics = exceeds.columns[exceeds.all()].to_list()
        affected_zones.update(df_configuration_data[df_configuration_data.CI_Name.str.strip() == 'Generic-System'][['Upstream_Dependencies','Downstream_Dependencies']].T.to_dict()[0])
        affected_zones['Performance Threshold Alerts'] = active_metrics
        affected_zones_str = "\n\n".join(f"{key}  :  {value}" for key, value in affected_zones.items())

        return affected_zones_str, active_metrics

    return {}, []


# Function to display relevant incidence and telemetry data
def create_master_data(input_text:str, search_mode):
    if search_mode == "Search by Description":
        # get the required details about the top incidence matches
        top_matches = get_nearest_match(input_text,top_n=50)
    elif search_mode == "Search by ID":
        selected_id_description = df_incidents[df_incidents['incident_id'].isin([input_text])]['description'].iloc[0]
        top_matches = get_nearest_match(selected_id_description, top_n=50)

    df_incident_filtered = df_incidents[df_incidents['incident_id'].isin(top_matches)]
    # Define custom order
    custom_order = ['Open', 'In Progress', 'Resolved', 'Closed']
    # Convert column to categorical with specified order
    df_incident_filtered['Category'] = pd.Categorical(df_incident_filtered['status'], categories=custom_order, ordered=True)
    # Sort DataFrame
    df_incident_filtered = df_incident_filtered.sort_values(by='Category')

    df_incident_display = df_incident_filtered[['incident_id','description','status','application','source','created_at','resolved_at']]
    df_incident_display = check_duplicate_incidents(df_incident_display)
    df_telemetry_display = df_incident_filtered[['incident_id','cpu_usage', 'memory_usage', 'disk_usage', 'network_latency']]

    affected_zones, active_metrics = affected_areas(df_incident_filtered, input_text)

    return df_incident_display, df_telemetry_display, affected_zones, active_metrics, df_incident_filtered


if __name__=="__main__":
    save_incidence_embeddings()