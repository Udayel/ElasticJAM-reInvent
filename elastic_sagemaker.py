import json
import os
import boto3
import botocore
import streamlit as st
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from elasticsearch import Elasticsearch


# AWS / SageMaker Settings

max_tokens=4000
max_context_tokens=4000
safety_margin=5

### Elastic Settings

#cluster Settings
cid = os.environ['ES_CLOUD_ID']
cp = os.environ['ES_PASSWORD']
cu = os.environ['ES_USERNAME']
es_model_id = 'multilingual-e5-base'
region=os.environ['AWS_REGION']


#Elasticsearch index
es_index = 'search-fiqa-ml'



# Get the SSM client
ssm_client = boto3.client('ssm',region_name=region)

# The name of the variable to write
variable_name = '/jam/elastic/task1status'

# The value of the variable
variable_value = 'Completed'


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

#truncate text
def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])



def llm_without_context(query, llm_model, endpoint_name):

    prompt = f"Answer this query in complete sentence. {query}"
    newline, bold, unbold = "\n", "\033[1m", "\033[0m"

    payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 200,
        "top_k":50,
        "top_p":0.95,
        "do_sample":True
        }
        }

    
    #print('prompt is: ',prompt)

    client = boto3.client("runtime.sagemaker",region_name=region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, InferenceComponentName=llm_model, ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions[0]["generated_text"]
    print(f"Input Text: {payload['inputs']}{newline}" f"Generated Text: {bold}{generated_text}{unbold}{newline}")

    ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String',Overwrite=True)
    st.markdown(f"AI: {generated_text.strip()}")

def llm_with_context(query, llm_model, endpoint_name):

    es = es_connect(cid, cu, cp)
    resp, url = search(query, es, es_index)
    resp = truncate_text(resp, max_context_tokens - max_tokens - safety_margin)
    prompt = f"Answer this question in complete sentence using only the information from this Elastic Doc: {resp}.question:{query}"
    newline, bold, unbold = "\n", "\033[1m", "\033[0m"

    payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 200,
        "top_k":50,
        "top_p":0.95,
        "do_sample":True
        }
        }


    client = boto3.client("runtime.sagemaker",region_name=region)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, InferenceComponentName=llm_model, ContentType="application/json", Body=json.dumps(payload).encode("utf-8")
    )
    model_predictions = json.loads(response["Body"].read())
    generated_text = model_predictions[0]["generated_text"]
    print(f"Input Text: {payload['inputs']}{newline}" f"Generated Text: {bold}{generated_text}{unbold}{newline}")

    ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String',Overwrite=True)
    st.markdown(f"AI: {generated_text.strip()}")




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

st.title("AI Assistant")


# Streamlit Form
st.markdown("""
        <style>
        .small-font {
            font-size:12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    endpoint_name = st.text_input('SageMaker Endpoint Name', '')
    llm_model = st.text_input('SageMaker Model Name', '')

st.markdown('<p class="small-font">Example:</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">How is taxation for youtube/twitch etc monetization handled in the UK?<br></p>', unsafe_allow_html=True)



with_context = st.toggle('With Context')
with st.form("chat_form"):
    query = st.text_input("What can I help you with: ")
    submit = st.form_submit_button("Submit")

# Generate and display response on form submission

if submit:
    if with_context:
        llm_with_context(query, llm_model,endpoint_name)
    else:
        llm_without_context(query, llm_model,endpoint_name)
