#!/usr/bin/env python3
import os
from qdrant_client import QdrantClient, models

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")

cli = QdrantClient(url=QDRANT_URL)

# Create keyword indexes for metadata fields
cli.create_payload_index(
    collection_name=COLLECTION,
    field_name="metadata.language",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
cli.create_payload_index(
    collection_name=COLLECTION,
    field_name="metadata.path_prefix",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

info = cli.get_collection(COLLECTION)
print(info.payload_schema)
