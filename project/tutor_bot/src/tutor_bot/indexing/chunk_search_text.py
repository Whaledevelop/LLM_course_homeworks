def build_chunk_search_text(
    note_title: str,
    heading_title: str,
    text: str,
) -> str:
    parts = [note_title]

    if heading_title and heading_title != note_title:
        parts.append(heading_title)

    parts.append(text)

    return "\n\n".join(parts)
