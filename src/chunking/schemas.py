from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel

ContentType = Literal["concept", "activity", "exercise", "general"]


class ChunkMetadata(BaseModel):
    class_num: int
    subject: str
    chapter_num: int
    chapter_name: str
    section_number: str = ""   # e.g. "2.1.3" ("" if none)
    section_title: str = ""
    activity_label: str = ""   # e.g. "Activity 2.5" ("" if not an activity)
    content_type: ContentType = "concept"
    strategy: Optional[str] = None  # e.g. "fixed", "recursive", "token", "semantic", "markdown", "hierarchical"


class Chunk(ChunkMetadata):
    chunk_id: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured metadata & size metrics about the chunk."""
        return {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "class_num": self.class_num,
            "subject": self.subject,
            "chapter_num": self.chapter_num,
            "chapter_name": self.chapter_name,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "activity_label": self.activity_label,
            "content_type": self.content_type,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
        }


class ParentChunk(ChunkMetadata):
    parent_id: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    def get_summary(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "strategy": self.strategy,
            "class_num": self.class_num,
            "subject": self.subject,
            "chapter_num": self.chapter_num,
            "chapter_name": self.chapter_name,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "content_type": self.content_type,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
        }


class ChildChunk(ChunkMetadata):
    chunk_id: str
    parent_id: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    def get_summary(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "strategy": self.strategy,
            "class_num": self.class_num,
            "subject": self.subject,
            "chapter_num": self.chapter_num,
            "chapter_name": self.chapter_name,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "content_type": self.content_type,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "line_count": self.line_count,
        }
