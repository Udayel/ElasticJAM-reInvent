import os
from elasticsearch import Elasticsearch
import boto3
import botocore
from pprint import pprint


### Elastic Settings

#cluster Settings
cid = os.environ['ES_CLOUD_ID']
cp = os.environ['ES_PASSWORD']
cu = os.environ['ES_USERNAME']
es_model_id = 'multilingual-e5-base'

es_index = 'search-fiqa-ml'
region=os.environ['AWS_REGION']

## SageMaker 

# Create a session with AWS
#session = boto3.Session()

# Get the SSM client
#ssm_client = session.client('ssm')
ssm_client = boto3.client('ssm',region_name=region)

# The name of the variable to write
variable_name = '/jam/elastic/task2status'

# The value of the variable
variable_value = 'Completed'


# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, basic_auth=(user, passwd))
    return es


# Search ElasticSearch index and return body and URL of the result
def search(query_text, index_name, es):
    
    print("Query text is", query_text)
    

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

    return body


es = es_connect(cid, cu, cp)

print('\n\n-----------------------------------------')
print("Enter Domain specific question \nexamples (Which colors can one use to fill out a check in the US?\nHow is taxation for youtube/twitch etc monetization handled in the UK?)")
query_text = input("Your Question > ")

body = search(query_text, es_index, es)
ssm_client.put_parameter(Name=variable_name, Value=variable_value, Type='String', Overwrite=True)
print("\n\nThe content from Elasticsearch is: ")
pprint(body)
