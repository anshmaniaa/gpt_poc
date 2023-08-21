# Description: Main script for ingesting documents into the database.
import argparse
from db.utils import VectorDB, ProcessDocument

def main(doc_id):
    
    vector_db = VectorDB()
    
    if doc_id not in vector_db.get_document_names():
        try:
            print(f"Document {doc_id} does not exist in the database.")
            vector_db.add_documents(
                ProcessDocument(f"protocols/{doc_id}.pdf").load_and_chunk()
            )
        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--doc_id', type=str, help='document id')
    args = argparse.parse_args()

    doc_id = args.doc_id
    if doc_id is not None:
        main(doc_id)
    else:
        print("Please provide a document id.")

    