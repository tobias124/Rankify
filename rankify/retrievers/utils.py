import json


def count_file_lines(path):
    """Efficiently count the number of lines in a file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count

def process_chunk(chunk_lines, start_idx, dense_index=False):
    """
    Process a chunk of lines from the corpus file and convert them to the required format.

    Args:
        chunk_lines (list): Lines from the corpus file.
        start_idx (int): Fallback starting index for IDs.
        dense_index (bool): True for dense format with separate title field.

    Returns:
        list: List of JSON strings with the required fields.
    """
    results = []
    for i, line in enumerate(chunk_lines):
        try:
            doc = json.loads(line.strip())

            raw_id = doc.get("id")
            #Todo: replace doc can be removed when str is supported as type
            doc_id = (
                int(raw_id.replace("doc", "")) if isinstance(raw_id, str) and raw_id.startswith("doc")
                else int(raw_id) if raw_id
                else start_idx + i
            )

            contents = (doc.get("contents") or doc.get("text", "")).strip()
            title = doc.get("title", contents[:100] if contents else "").strip()

            if dense_index:
                # Keep title and contents separate
                doc_dict = {"id": doc_id, "contents": contents}
                if title:
                    doc_dict["title"] = title
                results.append(json.dumps(doc_dict, ensure_ascii=False))
            else:
                # For sparse, combine title and contents
                if title:
                    full_contents = f"{title}\n{contents}"
                else:
                    full_contents = contents
                results.append(json.dumps({"id": doc_id, "contents": full_contents}, ensure_ascii=False))

        except (json.JSONDecodeError, ValueError):
            continue
    return results

def generate_chunks(file_obj, chunk_size=512):
            """
            Generator to yield chunks of lines from a file.

            Args:
                file_obj (file object): The file object to read from.
                chunk_size (int): The number of lines per chunk.
            Yields:
                tuple: A tuple containing a list of lines and the starting index of the chunk.
            """
            buffer = []
            start_idx = 0
            for line in file_obj:
                buffer.append(line)
                if len(buffer) >= chunk_size:
                    yield buffer[:], start_idx
                    buffer = []
                    start_idx += chunk_size
            if buffer:
                yield buffer, start_idx
