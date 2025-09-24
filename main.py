import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
import gradio as gr


# Load environment variables in a file called .env
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


# Connect to OpenAI, Anthropic and Google
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()


# A generic system message
system_message = "You are a helpful assistant"


def main() -> None:
    ### High level entry point ###
    view = gr.Interface(
        fn=select_model,
        inputs=[gr.Textbox(
            label="Your message:"), gr.Dropdown(
            ["gpt-4o-mini", "claude-3-5-haiku-latest", "gemini-1.5-pro"],
            label="Select model", value="claude-3-5-haiku-latest")],
        outputs=[gr.Markdown(label="Response:")],
        flagging_mode="never"
    )
    view.launch(inbrowser=True)

# remember to pass the model tooooo!


def select_model(prompt, model):
    if model == "gpt-4o-mini":
        result = call_gpt(prompt, model)
    elif model == "claude-3-5-haiku-latest":
        result = call_claude(prompt, model)
    elif model == "gemini-1.5-pro":
        result = call_gemini(prompt, model)
    else:
        raise ValueError("Unknown model")
    return result


def call_gpt(prompt: str, model: str):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def call_claude(prompt: str, model: str):
    response = claude.messages.create(
        model=model,
        system=system_message,
        messages=prompt,
        max_tokens=500
    )
    return response.content[0].text


def call_gemini(prompt: str, model: str):
    gemini = google.generativeai.GenerativeModel(
        model_name=model,
        system_instruction=system_message
    )
    response = gemini.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    main()
