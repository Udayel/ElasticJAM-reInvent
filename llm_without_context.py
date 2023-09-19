import json
import os
import boto3
import botocore
import streamlit as st
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler


# AWS / SageMaker Settings
flan_t5_endpoint_name = os.environ["FLAN_T5_ENDPOINT"]
aws_region = os.environ["AWS_REGION"]
max_tokens=4000
max_context_tokens=4000
safety_margin=5

### Elastic Settings

#cluster Settings
cid = os.environ['ES_CLOUD_ID']
cp = os.environ['ES_PASSWORD']
cu = os.environ['ES_USERNAME']
es_model_id = 'multilingual-e5-base'

# ES Datsets Options
#ES_DATASETS = {
#        'Elastic Documentation' : 'search-elastic-docs',
#        }

#LLM_LIST: List[str] = ["Flan-T5-XL"]
llm_model = 'Flan-T5-XL'

es_index = 'search-wikipedia-e5-multilingual'

## SageMaker 

# Create a session with AWS
session = boto3.Session()

# Get the SSM client
ssm_client = session.client('ssm')

# The name of the variable to write
variable_name = '/jam/elastic/task1status'

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



def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])


def toLLM(query,
        llm_model,
    ):

    prompt = f"Answer this question: {query}"
    print('prompt is: ',prompt)


    # Call LLM
    if llm_model == "Flan-T5-XL":
        endpoint_name = flan_t5_endpoint_name
        content_handler = ContentHandlerFlan()

        llm=SagemakerEndpoint(
                endpoint_name=endpoint_name,
                region_name=aws_region, 
                model_kwargs={"temperature":1, "max_length": max_tokens},
                content_handler=content_handler
        )

        answer = llm(prompt)
    else:
        answer = "Not available. Please select LLM"

    print("Answer is",answer)

    
    # Print respose
    st.markdown(f"AI: {answer.strip()}")


## Main
st.set_page_config(
     page_title="AI Assistant",
     page_icon="🧠",
#     layout="wide"
)


st.sidebar.markdown("""
 <style>
     [data-testid=stSidebar] [data-testid=stImage]{
         text-align: center;
         display: block;
         margin-left: auto;
         margin-right: auto;
         width: 100%;
     }
 </style>
 """, unsafe_allow_html=True)

st.title("ElasticAWSJam AI Assistant")

print("Selected LLM Model is:",llm_model)

# Streamlit Form
st.markdown("""
        <style>
        .small-font {
            font-size:12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="small-font">Example Searches:</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Which colors can one use to fill out a check in the US?<br>How is taxation for youtube/twitch etc monetization handled in the UK?<br></p>', unsafe_allow_html=True)
with st.form("chat_form"):
    query = st.text_input("What can I help you with: ")
    search_no_context = st.form_submit_button("Search Without Context")

# Generate and display response on form submission

if search_no_context:
    toLLM(query, llm_model)
