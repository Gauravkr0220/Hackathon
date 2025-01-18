import pathway as pw
import asyncio
import time

import io
import json
import logging
from typing import Callable, Coroutine, List, Tuple
from pathway.xpacks.llm.vector_store import VectorStoreServer
from utils.splitter import splitter_function
from utils.embedder import embed_function

import PyPDF2

logger = logging.getLogger(__name__)

def parser(data: str) -> List[Tuple[str, dict]]:
    try:
        text = data
        
        # metadata = {
        #     "title": reader.metadata.title if reader.metadata.title else "Unknown",
        #     "number_of_pages": len(reader.pages),
        # }

        # append text to test.txt
        # with open("test.txt", "a") as f:
        #     f.write(text)
        
        return [("data", {"data": text})]
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return [("", {"error": str(e)})]

table = pw.io.fs.read("RAG_CONTENT/review_papers/",
                    format="plaintext_by_file",
                    mode="static",
                    with_metadata=True,)

# pw.io.jsonlines.write(table, "./data/test.jsonl")


vector_store = VectorStoreServer(
    table,
    parser=parser,
    embedder=embed_function,
    splitter=splitter_function,
)
vector_store.run_server(host="127.0.0.1", port=8001, threaded=False)
pw.run()