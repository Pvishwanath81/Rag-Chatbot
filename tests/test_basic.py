from utils.text_splitter import get_chunk_stats


def test_empty_chunk_stats():
    result = get_chunk_stats([])
    assert result["total_chunks"] == 0
