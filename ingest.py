import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import init_embeddings_and_vector_store


def load_document():
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    return docs

def split_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")
    return all_splits

def load_doc_in_vector_store():
    docs = load_document()
    all_splits = split_document(docs)

    #Just for demonstration/tutorial of Query Analysis we'll add Section
    third = len(all_splits) // 3
    for i, doc in enumerate(all_splits):
        if i < third:
            doc.metadata["section"] = "beginning"
        elif i < 2 * third:
            doc.metadata["section"] = "middle"
        else:
            doc.metadata["section"] = "end"

    vector_stores = init_embeddings_and_vector_store()
    document_ids = vector_stores.add_documents(documents=all_splits)
    print(f"Added {len(document_ids)} documents to the vector store.")
    return vector_stores

if __name__ == "__main__":
    load_doc_in_vector_store()
