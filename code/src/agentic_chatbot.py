import os
import pandas as pd

import openai
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Dict

from registered_files import df_incidents, df_configuration_data, df_recommendations, df_kb_articles

# Initialize OpenAI Chat Model (GPT-4)
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Define Tool for Executing a Script (Simulated)
@tool
def run_script(script_name: str) -> str:
    """Simulates running a shell script and returns the outcome."""
    try:
        return f"Script '{script_name}' executed successfully (simulated)."
    except Exception as e:
        return f"Failed to execute script '{script_name}': {str(e)}"


@tool
def get_telemetry_data(incident_id: str, metric: str) -> str:
    """Retrieves telemetry data for a given incident ID and metric."""
    try:
        incident = df_incidents[df_incidents['incident_id'] == incident_id].iloc[0]
        value = incident.get(metric, "N/A")
        return f"Telemetry data for incident {incident_id}, metric {metric}: {value}"
    except IndexError:
        return f"Incident {incident_id} not found."


def get_prompt(incident_section: str,
               kb_section: str,
               recommendations_section: str,
               scripts_section: str) -> str:

    prompt = f"""
                Instructions:
                You are a GenAI assistant for platform support.
                
                Use ONLY the information provided in the sections above to:
                1. Summarize the likely root cause based on user query and recent incidents.
                2. If any information is asked in the given user query about telemetry data, recommendations, knowledge base articles, or RCA. Please summarize it.
                2. Suggest specific resolution steps using insights from the Knowledge Base and Recommendations.
                3. If a script is clearly recommended in the Recommendations section, call it using run_script("<script_name>"). Do not invent or assume any new scripts.
                4. Do not reference tools, scripts, or actions that are not explicitly listed in the prompt.
                
                Be concise and specific. Use all conclusions strictly on the provided context.
                
                Relevant Incidents:
                {incident_section or 'No similar incidents found.'}                
                
                Knowledge Base:
                {kb_section or 'No relevant KB articles found.'}
                
                Recommendations:
                {recommendations_section or 'No recommendations found.'}
                
                Scripts to Run:
                {scripts_section or 'No scripts to run.'}  
            """
    return prompt

def preprocess_data(user_input_, df_incidents, selected_incident_id, active_metrics):
    # Incident data
    incidents = df_incidents[df_incidents.status != 'Open'].to_dict(orient="records")
    incident_section = "\n".join(["\n  ".join([f"{col}: {i.get(col, 'N/A')}" for col in incidents[0].keys()])
                                  for i in incidents])

    # configuration data
    ci_name = df_incidents[df_incidents.incident_id == selected_incident_id].reset_index(drop=True)['CI_Affected'][0]
    config = df_configuration_data[df_configuration_data.CI_Name == ci_name].to_dict(orient="records")

    # KB Data
    kb_titles = list(set(i.get('KB Title') for i in incidents if i.get('KB Title')))
    kb_articles = df_kb_articles[df_kb_articles['Title'].isin(kb_titles)].to_dict(orient="records")
    kb_section = "\n".join([f"- {kb.get('Title', 'Untitled')}: {kb.get('Description', 'No description')}"
                            for kb in kb_articles
                            ])

    # recommendations data
    default_matrics = ['high load average', 'service not responding', 'high cache size', 'slow DB queries']
    active_metrics.append(default_matrics)

    triggered_recommendations = df_recommendations[df_recommendations["Metric"].isin(list(set(active_metrics)))].to_dict(orient="records")
    triggered_recommendations = [dict(t) for t in {tuple(sorted(d.items())) for d in triggered_recommendations}]
    recommendations_section = "\n".join([
                                        f"  - Metric: {rec.get('Metric', 'N/A')}, "
                                        f"Trigger: {rec.get('Threshold Trigger', 'N/A')}, "
                                        f"Recommendation: {rec.get('Recommendation', 'N/A')}, "
                                        f"Script: {rec.get('Script Name', 'N/A')}"
                                        for rec in triggered_recommendations
                                    ])

    scripts_to_run = {rec['Recommendation']: rec['Script Name'] for rec in triggered_recommendations}
    scripts_section = "\n".join([
                                f"  - {rec}: {script}"
                                for rec, script in scripts_to_run.items()  # Iterate through both keys and values
                            ])

    return get_prompt(incident_section, kb_section, recommendations_section, scripts_section)

# Run Agent using LangChain
def run_genai_agent(user_input_: str,
                    df_incident_filtered: pd.DataFrame,
                    selected_incident_id: str,
                    active_metrics: list) -> str:
    prompt = preprocess_data(user_input_, df_incident_filtered, selected_incident_id, active_metrics)
    tools = [run_script, get_telemetry_data]

    # Create a ChatPromptTemplate for the agent
    agent_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt),  # Customize as needed
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        HumanMessage(content="{input}")
    ])

    agent = create_openai_functions_agent(llm=llm, prompt=agent_prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke({"input": user_input_})
    return result['output']