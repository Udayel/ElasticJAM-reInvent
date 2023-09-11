import os
from elasticsearch import Elasticsearch
import boto3

cid = os.environ['ES_CLOUD_ID']
cu = os.environ['ES_USERNAME']
cp = os.environ['ES_PASSWORD']

# Create a session with AWS
session = boto3.Session()

# Get the SSM client
ssm_client = session.client('ssm')

# The name of the variable to write
variable_name = '/jam/elastic/task3status'

# The value of the variable
variable_value = 'Completed'



# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
    return es


# Elastic Hybrid search query
def search(query_text):

    es = es_connect(cid, cu, cp)

    # Textual search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }]
        }
    }


    # Vector search
    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-elastic-docs'
    
    #hybrid search
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    
    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

query_text = input("Enter Domain specific question: ")
body, url = search(query_text)
ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String')
print("The content is: ", body)
print("The source URL is: ", url)
