"""Character windows that retain page boundaries and source offsets."""
from dataclasses import dataclass


@dataclass
class Chunk:
    page_content: str
    metadata: dict


def split_pages(pages, file_id, chunk_size=800, overlap_size=120):
    if not isinstance(chunk_size, int) or not 1 <= chunk_size <= 20000:
        raise ValueError("chunk_size must be between 1 and 20000 characters")
    if not isinstance(overlap_size, int) or not 0 <= overlap_size < chunk_size:
        raise ValueError("overlap_size must be nonnegative and smaller than chunk_size")
    chunks = []
    for page_index, page in enumerate(pages):
        text = page.page_content or ""
        for start in range(0, len(text), chunk_size - overlap_size):
            end = min(start + chunk_size, len(text))
            value = text[start:end]
            if value.strip():
                metadata = dict(page.metadata or {})
                metadata.update(file_id=file_id, page_index=page_index,
                                char_start=start, char_end=end,
                                chunk_strategy="page_overlap_v1")
                chunks.append(Chunk(value, metadata))
            if end == len(text):
                break
    return chunks
