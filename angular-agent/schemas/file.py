
from pydantic import BaseModel, Field


class FileGenerated(BaseModel):
    """Schema for code generation."""
    path: str = Field("The path where the file was generated")
    content: str = Field("The code generated")
