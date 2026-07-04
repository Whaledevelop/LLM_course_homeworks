from pydantic import BaseModel


class NoteTitleMatch(BaseModel):
    matched_title: str | None = None
