from typing import Literal
from pydantic import model_validator
from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0
    chunk_strategy: Literal["legacy", "page_overlap"] = "legacy"

    @model_validator(mode="after")
    def validate_chunk_window(self):
        if self.chunk_strategy == "page_overlap":
            if self.chunk_size is None or not 1 <= self.chunk_size <= 20000:
                raise ValueError("chunk_size must be between 1 and 20000")
            if self.overlap_size is None or not 0 <= self.overlap_size < self.chunk_size:
                raise ValueError("overlap_size must be smaller than chunk_size")
        return self
