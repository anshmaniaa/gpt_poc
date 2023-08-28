Here we implement a GPT-3.5 based system to extract information from protocol documents based on prompts for 47 variables. 

Prompts:
The prompts are divided in three parts:
1. System
The system prompt is the instruction given to GPT on how to process the prompt and what the output should look like. It also describes briefly the problem being solved to give more context to GPT.
2. User
The user prompt is mainly the query.
3. Context
The context is made up of pieces of text 

The system has two main components.

1. Ingestion
    1. Read
    2. Preprocess 
    3. Chunk
    4. Chunk quality check
    5. Embedd qualified chunks
    6. Write chunks and embeddings to DB
2. Prediction
    1. Get doc_id
    2. Read prompts 
    3. For every prompt retrieve appropriate context
    4. Add context to prompt and send to LLM
    5. Write extracted values to DB

# Ingestion
- Input: Document ID accepted from CLI as --doc_id

- Output: Chunks with metadata and their embeddings

Tables:


## Table structure and schema

### langchain_pg_collection


| name | cmetadata | uuid |
|------|----------|------|
|varchar|json|uuid|

### langchain_pg_embedding

| collection_id | embedding | document | cmetadata | custom_id | uuid |
|---------------|-----------|----------|-----------|-----------|------|
|uuid|vector|varchar|json|custom_id|uuid|


# Prediction
- Input: Document ID accepted from CLI as --doc_id

- Output: 47 pieces of extracted information as per Prompts

Tables:
### Prompts

| variable_id | prompt_id | system_message | user_message |
|---------------|-----------|----------|----------|
|varchar|uuid|varchar|varchar|

### GT

| doc_id | prompt_id | variable_id | gt_value | uuid |                             
|---------------|-----------|----------|-----------|-----------|
|uuid|vector|varchar|json|uuid|

