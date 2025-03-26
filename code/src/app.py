import gradio as gr
import pandas as pd
from controller import llm_response, create_master_data, unique_incident_ids, unique_application_ids
import plotly.express as px
from agentic_chatbot import run_genai_agent


# Predefined lists for Search by ID and Search by Application
search_by_id_options = ["Select an ID"] + unique_incident_ids
search_by_application_options = ["Select an Application"] + unique_application_ids


def update_input(search_mode):
    """Show the appropriate input field based on search mode and reset previous input values"""
    if search_mode == "Search by ID":
        return (
            gr.update(visible=True, choices=search_by_id_options, value=None),  # Reset dropdown
            gr.update(visible=False, value="")  # Hide textbox & reset it
        )
    elif search_mode == "Search by Application":
        return (
            gr.update(visible=True, choices=search_by_application_options, value=None),  # Reset dropdown
            gr.update(visible=False, value="")  # Hide textbox & reset it
        )
    elif search_mode == "Search by Description":
        return (
            gr.update(visible=False, value=None),  # Hide dropdown & reset it
            gr.update(visible=True, value="")  # Show textbox & reset it
        )
    else:
        return (
            gr.update(visible=False, value=None),  # Hide dropdown & reset
            gr.update(visible=False, value="")  # Hide textbox & reset
        )

def generate_chart(df):
    # Convert DataFrame to long format
    df_long = df.melt(id_vars=["incident_id"], var_name="Category", value_name="Value")

    # Create line plot using Plotly
    fig = px.line(df_long, x="incident_id", y="Value", color="Category", markers=True,
                  title="Telemetry chart")

    # Rotate x-axis labels by 90 degrees
    fig.update_layout(xaxis=dict(tickangle=90))

    return fig


# Fetch master data
def generate_gradio_data(dropdown_value, textbox_value, search_mode):
    if search_mode == "Search by ID" and dropdown_value in ["", "Select an ID"]:
        return [], None
    elif search_mode == "Search by Application" and dropdown_value in ["", "Select an Application"]:
        return [], None
    elif search_mode == "Search by Description" and textbox_value.strip() == "":
        return [], None

    input_value = dropdown_value if search_mode in ["Search by ID", "Search by Application"] else textbox_value
    df_incident_display, df_telemetry_display, affected_zones, active_metrics, df_incident_filtered = create_master_data(input_value, search_mode)

    return df_incident_display, generate_chart(df_telemetry_display), affected_zones


# Function to reset inputs and outputs
def reset_inputs():
    return gr.update(value=None, visible=False), gr.update(value="", visible=False), gr.update(value=""), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), [], []


def chatbot_response(user_message, chatbot_history, dropdown_value, search_mode):
    df_incident_display, df_telemetry_display, affected_zones, active_metrics, df_incident_filtered = create_master_data(dropdown_value, search_mode)

    if chatbot_history is None:  # Ensure chatbot_history is initialized
        chatbot_history = []

    # Append user message
    chatbot_history.append({"role": "user", "content": user_message})

    # Generate bot response (simple echo for demo)
    bot_response = run_genai_agent(user_message, df_incident_filtered, dropdown_value, active_metrics)
    # bot_response = llm_response(chatbot_history, chatbot_activation=True)
    bot_message = {"role": "assistant", "content": bot_response}

    # Append bot response
    chatbot_history.append(bot_message)

    return chatbot_history, gr.update(value="")  # Clear input box after response


# Create Gradio interface
with gr.Blocks() as demo:
    # Title across the top of the page with background color and centered
    gr.Markdown(
        "# Welcome to Integrated Platform powered by SWAT Genie",
        elem_id="title"
    )

    with gr.Row():  # Create a horizontal layout row for left and right columns
        with gr.Column(scale=1):  # Left column for image display (1/4 of space)
            # Image without bounding box
            image = gr.Image(value=r"./data/logo.jpg",
                             label=None, interactive=False, show_label=False)

        with gr.Column(scale=3):  # Right column for input box and buttons (3/4 of space)
            with gr.Row():
                search_mode = gr.Dropdown(
                    label="Select Search Mode",
                    choices=["","Search by ID", "Search by Application", "Search by Description"],
                    value=""
                )
                dropdown_input_value = gr.Dropdown(choices=[], label="Select Option", visible=True, interactive=True)
                text_input_value = gr.Textbox(label="Enter Description", placeholder="Type here...", visible=False,
                                           interactive=True)

                search_mode.change(update_input, inputs=[search_mode], outputs=[dropdown_input_value,text_input_value])

            with gr.Row():  # Place buttons side by side
                submit_btn = gr.Button("Submit", size="sm")  # Small size for button
                clear_btn = gr.Button("Clear", size="sm")  # Small size for button

    # Below the input/output section, create another row for the results
    with gr.Row():
        with gr.Column():  # Left column for tabular result
            table_output = gr.DataFrame(label="Similar Incidence", visible=False)

        with gr.Column():  # Right column for text result
            with gr.Group(visible=False) as chatbot_section:
                with gr.Column(elem_id="chatbot-container"):
                    chatbot = gr.Chatbot(label="Chatbot", type='messages')
                    user_input = gr.Textbox(placeholder="Type a message...", label="Chatbot Input")
            # chart = gr.Plot(label="Telemetry Graph", visible=False, elem_id="custom-chart")

    with gr.Row():
        with gr.Column(scale=2):
            chart = gr.Plot(label="Telemetry Graph", visible=False, elem_id="custom-chart")

        with gr.Column(scale=2):
            affected_zones_display = gr.Textbox()

    submit_btn.click(fn=generate_gradio_data, inputs=[dropdown_input_value,text_input_value,search_mode], outputs=[table_output, chart, affected_zones_display])

    # After processing, show the results
    submit_btn.click(
        fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)],
        inputs=None,
        outputs=[table_output, chart, chatbot_section])

    # Use gr.State to manage the chat history
    chatbot_history = gr.State([])  # Initialize empty list for chatbot history

    # Clear button functionality
    clear_btn.click(fn=reset_inputs, inputs=None, outputs=[dropdown_input_value,text_input_value,search_mode,table_output, chart, chatbot_section, chatbot, chatbot_history])

    # When user submits a message in the chatbot input, show the chatbot and update the history
    user_input.submit(fn=chatbot_response, inputs=[user_input, chatbot_history, dropdown_input_value, search_mode], outputs=[chatbot, user_input])

    # Custom CSS for styling (using HTML component)
    gr.HTML("""
        <style>
            #title {
                text-align: center;  /* Center the title */
                background-color: #6495ED;  /* Green background color */
                color: white;  /* White text */
                padding: 20px;  /* Add padding around the title */
                font-size: 32px;  /* Larger font size */
                border-radius: 10px;  /* Optional: Rounded corners */
            }
            #custom-chart {
            height: 450px !important;  /* Adjust height as needed */
            }
            #chatbot-container {
                background-color: #7FFFD4;
                border-top-left-radius: 10px;
                padding: 10px;
                box-shadow: -2px -2px 10px rgba(0, 0, 0, 0.1);
                overflow-y: auto;
            }
        </style>
    """)


if __name__=="__main__":
    # Launch the app
    demo.launch()