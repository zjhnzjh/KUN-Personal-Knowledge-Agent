from app.rag import ParsedSection, chunk_sections


def test_chunking_preserves_locator_and_overlap():
    text = "第一段。" * 200
    chunks = chunk_sections([ParsedSection("第 1 页", "测试", text)], size=120, overlap=20)
    assert len(chunks) > 1
    assert all(item.locator == "第 1 页" for item in chunks)
    assert all(item.text for item in chunks)
