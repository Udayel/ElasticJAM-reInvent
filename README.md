# ElasticJAM-reInvent
Elastic-AWS Jam

## Run the chatbot application and ask a domain specific question

1. Create SageMaker Domain: Navigate to Amazon SageMaker. Click on Getting Started. Click on Create SageMaker Domain. Select Quick setup
2. Deploy LLM: Navigate to Studio. You should see user profile populated. Click Open Studio.
Navigate to SageMaker JumpStart and select Models, notebooks, solutions. In the search window, search for Flan-T5 XL. Click Deploy. Copy the Endpoint name and update it in config.sh
3. Login to your Amazon EC2 instance
<br />Instance> Connect> EC2 Instance Connect> Connect
4. Install pip <br /> Command: ```sudo yum install python-pip```
5. Clone the git repo <br /> Command: ```git clone https://github.com/Udayel/ElasticJAM-reInvent.git```
6. Navigate to the application directory.<br />Command : ```cd RAGElastic-LLM```
7. Install dependencies <br /> Command: ```pip install -r requirements.txt```
8. Open config.sh and update the following configurations <br />**ES_CLOUD_ID**: Elastic Cloud Deployment ID<br />**ES_USERNAME**: Elasticsearch Cluster User<br />**ES_PASSWORD**: Elasticsearch User password<br />**FLAN_T5_ENDPOINT**: Amazon SageMaker Endpoint Name<br />
9. Run the application <br />Command: ```streamlit run llm_without_context.py```
10. Open the External URL which popped up in the console
11. Type the following question and submit. <br />Question: *What is Elastic?*

## Retrieve Relevant Context from Elasticsearch

1. Open the file **elastic_hybrid_search_query.py** and check the hybrid search query
2. Run the application<br />
Command: ```python3 elastic_hybrid_search.py```
3. Type the domain specific question you asked in Task 1 and check the response

## Retrieval Augmented Generation(RAG) using context from Elasticsearch

1. Open the file **llm_with_context.py** and check the application code
2. Run the application<br />
Command: ```streamlit run llm_with_context.py```
3. Open the **External URL** in the browser
3. Type the domain specific question you asked in Task 1 and check the response
