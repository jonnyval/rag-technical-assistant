import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
from langchain_community.storage import SQLStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings  # noqa: E402


PARENT_NAMESPACE = "reglab_parents"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test ticket retrieval without LLM.")
    parser.add_argument(
        "query",
        nargs="?",
        default="контроллер не виден в сети после обновления прошивки",
        help="Search query.",
    )
    parser.add_argument("--db", default=None, help="Backend name. Defaults to second_db.")
    parser.add_argument("--k", type=int, default=5, help="Number of results.")
    parser.add_argument("--child", action="store_true", help="Show raw child chunks instead of parent docs.")
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_backend(name: str) -> dict[str, Any]:
    backend = settings.db_backends.get(name)
    if not backend:
        raise SystemExit(f"Backend not found: {name}")
    return backend


def parent_store_path(backend: dict[str, Any]) -> Path:
    parent_store = backend.get("parent_store", {})
    if isinstance(parent_store, dict):
        path_value = parent_store.get("path", "vector_dbs/parent_docstore.db")
    else:
        path_value = parent_store or "vector_dbs/parent_docstore.db"
    return resolve_project_path(str(path_value))


def make_client(backend: dict[str, Any]) -> QdrantClient:
    if backend.get("url"):
        return QdrantClient(url=backend["url"])
    if backend.get("path"):
        return QdrantClient(path=str(resolve_project_path(backend["path"])))
    raise SystemExit("Backend must define url or path.")


def build_retriever(backend_name: str, k: int) -> tuple[ParentDocumentRetriever, QdrantClient, str]:
    backend = get_backend(backend_name)
    collection = backend["collection"]
    client = make_client(backend)
    if not client.collection_exists(collection):
        raise SystemExit(f"Qdrant collection not found: {collection}")

    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Instruct: Retrieve relevant technical support ticket to answer the query.\nQuery: ",
        },
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    store_path = parent_store_path(backend)
    if not store_path.exists():
        raise SystemExit(f"Parent store not found: {store_path}")
    byte_store = SQLStore(namespace=PARENT_NAMESPACE, db_url=f"sqlite:///{store_path}")
    docstore = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda key: key,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )
    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": k},
    )
    return retriever, client, collection


def preview(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def main() -> None:
    args = parse_args()
    backend_name = args.db or settings.second_db_name
    retriever, client, collection = build_retriever(backend_name, args.k)
    info = client.get_collection(collection)

    print(f"Backend: {backend_name}")
    print(f"Collection: {collection}")
    print(f"Qdrant points: {info.points_count}")
    print(f"Query: {args.query}")
    print()

    if args.child:
        docs = retriever.vectorstore.similarity_search(args.query, k=args.k)
    else:
        docs = retriever.invoke(args.query)

    if not docs:
        print("No results.")
        return

    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"--- #{index}")
        print(f"ticket_id: {meta.get('ticket_id')}")
        print(f"title: {meta.get('page_title')}")
        print(f"equipment: {meta.get('equipment_type')}")
        print(f"category: {meta.get('category')}")
        print(f"url: {meta.get('ticket_url')}")
        print(preview(doc.page_content))
        print()


if __name__ == "__main__":
    main()
