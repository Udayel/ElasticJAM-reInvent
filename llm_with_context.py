import boto3
import botocore
import json
import os
import streamlit as st
from elasticsearch import Elasticsearch
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

llm_model = 'Flan-T5-XL'
es_index = 'search-fiqa-ml'
region=os.environ['AWS_REGION']

## SageMaker 

# Create a session with AWS
#session = boto3.Session()

# Get the SSM client
#ssm_client = session.client('ssm')
ssm_client = boto3.client('ssm',region_name=region)

# The name of the variable to write
variable_name = '/jam/elastic/task3status'

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


# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    print("Connection is", es)
    return es


# Search ElasticSearch index and return body and URL of the result
def search(query_text, es, index_name):

    
    '''Using ELSER -
       Search ElasticSearch index and return body and URL of the result'''

    query = {
      "bool": {
        "should": [
          {
            "match": {
              "text": {
                "query": query_text,
                "boost": 0.01
              }
            }
          },
          {
            "text_expansion": {
              "ml.inference.text_expanded.predicted_value": {
                "model_id": ".elser_model_1",
                "model_text": query_text
              }
            }
          }
        ]
      }
    }


    fields = ["text", 
             ]

    index = index_name
    resp = es.search(index=index,
                     query=query,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['text'][0]

    url=''

    return body, url


def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])


def toLLM(query,
        llm_model,
        index=False
    ):

    # Set prompt and add ES contest if required
    if index:
        es = es_connect(cid, cu, cp)
        resp, url = search(query, es, index)
        resp = truncate_text(resp, max_context_tokens - max_tokens - safety_margin)
        prompt = f"Answer this question in complete sentence using only the information from this Elastic Doc: {resp}.question:{query}"
        with st.expander("Source Document From Elasticsearch"):
            st.markdown(resp)
    else:
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
    if index:
        if negResponse in answer:
            st.markdown(f"AI: {answer.strip()}")
        else:
            ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String',Overwrite=True)
            st.markdown(f"`AI`: {answer.strip()}")
    else:
        st.markdown(f"`AI`: {answer.strip()}")


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
    search_context = st.form_submit_button("Search With Context")


# Generate and display response on form submission
negResponse = "I'm unable to answer the question based on the information I have from Context."


if search_context:
    toLLM(query, llm_model, es_index)

