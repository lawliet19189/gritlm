from enum import Enum


class PromptType(Enum):
    """Enumerated class to select the prompt type"""

    NO_RETRIEVAL = "no_retrieval"
    FULL_FORMAT = "full_format"
    FULL_FORMAT_NO_EMBED = "full_format_no_embed"
    FULL_FORMAT_DOC = "full_format_doc"
    FULL_FORMAT_NO_EMBED_DOC = "full_format_no_embed_doc"
    CACHE_FORMAT_QUERY = "cache_format_query"
    CACHE_FORMAT_DOC = "cache_format_doc"
    CACHE_FORMAT_DOC_QUERY = "cache_format_doc_query"
    CACHE_FORMAT_QUERY_DOC = "cache_format_query_doc"
