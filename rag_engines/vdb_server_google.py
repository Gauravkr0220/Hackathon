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


def pdf_parser(file_bytes: bytes) -> List[Tuple[str, dict]]:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        
        # metadata = {
        #     "title": reader.metadata.title if reader.metadata.title else "Unknown",
        #     "number_of_pages": len(reader.pages),
        # }
        # append text to test.txt
        # with open("test.txt", "a") as f:
        #     f.write(text)
        
        return [(text, {})]
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return [("", {"error": str(e)})]

table = pw.io.gdrive.read(
    object_id="14CZj0xiGScJ9Q2v7xRmLpAqOHEUN8H23",
    service_user_credentials_file="kdsh-pathway-bab3103d5539.json",
    mode="static",
    with_metadata=True
)

pw.io.jsonlines.write(table, "./data/test.jsonl")


# vector_store = VectorStoreServer(
#     table,
#     parser=pdf_parser,
#     embedder=embed_function,
#     splitter=splitter_function,
# )
# vector_store.run_server(host="127.0.0.1", port=8000, threaded=False)
pw.run()