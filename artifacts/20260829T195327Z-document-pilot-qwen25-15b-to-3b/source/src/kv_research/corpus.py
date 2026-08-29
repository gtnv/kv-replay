import hashlib
import re

import torch
from datasets import load_dataset


def dataset_documents(stream, text_field):
    texts = []
    source_rows = []
    document_index = 0
    for row_index, row in enumerate(stream):
        if text_field not in row:
            raise RuntimeError(f"dataset row {row_index} lacks required field {text_field}")
        text = row[text_field]
        if not isinstance(text, str):
            raise RuntimeError(f"dataset row {row_index} field {text_field} is not text")
        title = re.fullmatch(r"= [^=].* =", text.strip()) is not None
        if title and texts:
            yield {
                "document_index": document_index,
                "document_start_row": source_rows[0],
                "source_rows": source_rows,
                "text": "".join(texts),
            }
            document_index += 1
            texts = []
            source_rows = []
        if title or (texts and text.strip()):
            texts.append(text)
            source_rows.append(row_index)
    if texts:
        yield {
            "document_index": document_index,
            "document_start_row": source_rows[0],
            "source_rows": source_rows,
            "text": "".join(texts),
        }


def token_chunks(
    tokenizer,
    dataset_id,
    dataset_config,
    revision,
    split,
    text_field,
    length,
    count,
    fold_modulus,
    fold_residue,
    minimum_source_row,
):
    if not 0 <= fold_residue < fold_modulus:
        raise RuntimeError(f"invalid document fold {fold_residue}/{fold_modulus}")
    stream = load_dataset(
        dataset_id,
        dataset_config,
        revision=revision,
        split=split,
    )
    chunk_index = 0
    for document in dataset_documents(stream, text_field):
        if document["document_start_row"] < minimum_source_row:
            continue
        document_hash = hashlib.sha256(document["text"].encode()).hexdigest()
        if int(document_hash[:16], 16) % fold_modulus != fold_residue:
            continue
        values = tokenizer(document["text"], add_special_tokens=False)["input_ids"]
        values.append(tokenizer.eos_token_id)
        if len(values) < length:
            continue
        values = values[:length]
        token_bytes = torch.tensor(values, dtype=torch.int32).numpy().tobytes()
        yield {
            "dataset_split": split,
            "chunk_index": chunk_index,
            "document_index": document["document_index"],
            "document_start_row": document["document_start_row"],
            "document_sha256": document_hash,
            "source_rows": document["source_rows"],
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "input_ids": torch.tensor(values, dtype=torch.long),
        }
        chunk_index += 1
        if chunk_index == count:
            return
    raise RuntimeError(f"dataset ended after {chunk_index} chunks; required {count}")
