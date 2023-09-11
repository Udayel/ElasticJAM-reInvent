from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
import os
import streamlit as st
from elasticsearch import Elasticsearch
from typing import List, Tuple, Dict
from langchain.llms.bedrock import Bedrock
import boto3
import botocore

#Environment variables
flan_t5_endpoint_name = os.environ["FLAN_T5_ENDPOINT"]
aws_region = os.environ["AWS_REGION"]

max_tokens=1024
max_context_tokens=4000
safety_margin=5

LLM_LIST: List[str] = ["Flan-T5-XL"]

# Create a session with AWS
session = boto3.Session()

# Get the SSM client
ssm_client = session.client('ssm')

# The name of the variable to write
variable_name = '/jam/elastic/task2status'

# The value of the variable
variable_value = 'Completed'

class ContentHandlerFlan(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["generated_texts"][0]



st.title("Anycompany AI Assistant")

with st.sidebar.expander("⚙️", expanded=True):
    llm_model = st.selectbox(label='Large Language Model', options=LLM_LIST)

endpoint_name = flan_t5_endpoint_name
content_handler = ContentHandlerFlan()



# Main chat form
with st.form("chat_form"):
    query = st.text_input("You: ")
    submit_button = st.form_submit_button("Send")


# Generate and display response on form submission
negResponse = "I'm unable to answer the question"
if submit_button:
    prompt_without_context = f"Answer this question: \n"

    prompt = prompt_without_context


    llm=SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=aws_region, 
                model_kwargs={"temperature":1, "max_length": 500},
                content_handler=content_handler
            )

    answer = llm(prompt)
    print("Answer is",answer)

    ####stopping here
    
    if negResponse in answer:
        st.write(f"AI: {answer.strip()}")
    else:
        ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String')
        st.write(f"AI: {answer.strip()}")