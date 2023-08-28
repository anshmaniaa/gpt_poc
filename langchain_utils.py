import os
import openai
from dotenv import load_dotenv
from utils import SimilaritySearch
load_dotenv()

def search_documents(query, directory):
    """Search for documents in a directory that contain the query"""
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                text = f.read()
                similarity_search = SimilaritySearch()
                context = similarity_search.get_context(query, document_name=filename)
                if context:
                    results.append((filename, context))
    return results

def langchain_agent(query):
    """Use OpenAI to generate a response to a query"""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        engine='gpt-3.5-turbo',
        prompt=f"Search for documents that contain '{query}'\n\nResponse:",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    query = input("Enter a search query: ")
    results = search_documents(query, 'documents/')
    if results:
        print(f"Found {len(results)} documents that contain '{query}':")
        for result in results:
            print(f"{result[0]}:\n{result[1]}\n")
    else:
        print(f"No documents found that contain '{query}'")
    response = langchain_agent(query)
    print(f"Langchain agent response: {response}")