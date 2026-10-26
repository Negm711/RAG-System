from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5

from pydantic import Field, model_validator


class RerankedSearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    candidate_limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_limits(self):
        if not self.text.strip():
            raise ValueError("Question cannot be blank")
        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be at least limit")
        return self
