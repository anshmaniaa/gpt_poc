from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
import hashlib
import openai
import os
import langchain
import yaml
import sys
from db.utils import VectorDB

sys.path.append('..')

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )

langchain.llm_cache = GPTCache(init_gptcache)

#read prompts from yaml file 
def read_prompts_from_yaml(yaml_file):
    print(yaml_file)
    with open(yaml_file) as file:
        prompts = yaml.safe_load(file)
    return prompts

def main():
    vector_db =VectorDB(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    prompts = read_prompts_from_yaml("prediction/prompts.yml")
    vector_db.get_compose_prompt("Is biopsy done as part of this study?")
    extracted_values = llm.predict(prompt=prompts["assessment_level"], max_tokens=1, stop="\n")
    print(prompts["assessment_level"])
if __name__=="__main__":
    main()