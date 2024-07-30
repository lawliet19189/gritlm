from enum import Enum


class PromptType(Enum):
    """Enumerated class to select the prompt type"""

    NO_RETRIEVAL: str
    FULL_FORMAT: str
    FULL_FORMAT_NO_EMBED: str
    FULL_FORMAT_DOC: str
    FULL_FORMAT_NO_EMBED_DOC: str
    CACHE_FORMAT_QUERY: str
    CACHE_FORMAT_DOC: str
    CACHE_FORMAT_DOC_QUERY: str
    CACHE_FORMAT_QUERY_DOC: str
