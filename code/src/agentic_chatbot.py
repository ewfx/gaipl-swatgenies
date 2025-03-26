import os
import json
import pandas as pd

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent,AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

from registered_files import df_incidents, df_configuration_data, df_recommendations, df_kb_articles

openai.api_key = os.getenv("OPENAI_API_KEY")
class LLMAGENT():
    def __init__(self):
    # Initialize OpenAI Chat Model (GPT-4)
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def get_prompt(self, selected_incident_id: str,
                   incident_section: str,
                   kb_section: str,
                   recommendations_section: str,
                   scripts_section: str) -> str:

        prompt = f"""
                    Instructions:
                    You are a GenAI assistant for platform support.
                    
                    Use ONLY the information provided in the sections above to:
                    1. Summarize the likely root cause based on user query and recent incidents.
                    2. If any information is asked in the given user query about telemetry data, recommendations, knowledge base articles, or RCA. Please summarize it.
                    3. If a script is clearly recommended in the Scripts to run section, call it using run_script tool defined and resolve the incident. Do not invent or assume any new scripts.
                    4. If relevant incident information is not found, respond with: "The incident selected does not have relevant information. Please select the incident related to your query."
                    5. Do not reference tools, scripts, or actions that are not explicitly listed in the prompt.
                    
                    Be concise and specific. Use all conclusions strictly on the provided context.
                    Incident ID:
                    {selected_incident_id or 'No incidents found'}
                    
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

    def preprocess_data(self, df_incidents, selected_incident_id, active_metrics):
        # Incident data
        # incidents = df_incidents[df_incidents.status != 'Open'].to_dict(orient="records")
        incidents = df_incidents.to_dict(orient="records")
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
        active_metrics.extend(default_matrics)

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
        agent_prompt = self.get_prompt(selected_incident_id, incident_section, kb_section, recommendations_section, scripts_section)

        return agent_prompt, scripts_to_run

    def llm_query_tool(self, incident_chain, query):
        # Assuming data_dict is defined earlier with the necessary keys
        # Pass individual keys to incident_chain.run, ensuring all expected variables are present
        response = incident_chain.run(query=query, chat_history=self.memory.load_memory_variables({})['chat_history'])
        return response

    def run_script(self, script):
        # script_lookup = {v: k for k, v in scripts_to_run.items()}
        # if script not in script_lookup:
        #     return f"✅ {script} ran succesfully"
        # try:
        #     return f"{script_lookup[script]} - is completed"
        # except Exception as e:
        return f"{script} executed successfully"

    # LLM setup
    def get_tools(self, incident_chain):
        # Tools for the agent
        tools = [
            Tool(
                name="Incident Query",
                func=lambda query: self.llm_query_tool(incident_chain, query),
                description="Use LLM to interpret the incident data and answer any query."
            ),
            Tool(
                name="Run Script",
                func=lambda script: self.run_script(script),
                description="Use LLM to interpret the incident data and run the script to resolve."
            )
        ]

        return tools


    # Run Agent using LangChain
    def run_genai_agent(self, user_input_: str,
                        df_incident_filtered: pd.DataFrame,
                        selected_incident_id: str,
                        active_metrics: list,
                        ) -> str:
        prompt, scripts_to_run = self.preprocess_data(df_incident_filtered, selected_incident_id, active_metrics)

        llm_prompt = PromptTemplate(
                                    input_variables=["query"],
                                    template="User Query:\n{query}\n\n" + prompt
                                   )

        # LLM chain
        incident_chain = LLMChain(llm=self.llm, prompt=llm_prompt)
        tools = self.get_tools(incident_chain)

        # Initialize the interactive agent
        agent = initialize_agent(
                                tools,
                                self.llm,
                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                verbose=True,
                                memory=self.memory,
                                handling_parsing_errors=True
                                )

        result = agent.run(f"{user_input_}")
        return result