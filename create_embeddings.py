"""
Methods for creating embeddings from text
"""

import os
import boto3
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import pdftotext as ptt
import tiktoken  # for counting tokens
from config import (
    EMBEDDING_MODEL,
    ANSWERING_MODEL,
    MAX_TOKENS,
    BATCH_SIZE,
    BASE_DIR,
    PDF_TEXT_FILE_PATH,
    DEBUG,
    EMBEDDINGS_SAVE_PATH
)
from ai_compendium import mongo_collection


def generate_embeddings(source_file_path, doc_id):
    """
    steps:
    1. download file from source_file_path
    2. extract text from downloaded file (pdf2text)
    3. create embeddings data from extracted text file
    4. Save embeddings data to db
    5. return embedding file path

    :return:
    """
    #  write functionality here
    text_data = convert_to_text(source_file_path, doc_id, save_text_file=False)
    save_path = create_embeddings_from_text(text_data, doc_id)
    return save_path


def convert_to_text(file_path: str, doc_id: str, save_text_file: bool = False):
    """
    convert provided file to text file
    :param file_path:
    :param doc_id:
    :param save_text_file:
    :return:
    """
    with open(file_path, "rb") as f:
        pdf = ptt.PDF(f)
    # Read all the text into one string
    pdf_text = "\n\n".join(pdf)
    print("Completed text extraction from pdf file")
    # write text file
    os.remove(file_path)
    print(f"Deleted pdf file from: {file_path}")
    if save_text_file:
        save_dir = "./artifacts/pdf_text/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{doc_id}_pdf2text.txt")
        with open(save_path, "w") as text_file:
            text_file.write(pdf_text)
    return pdf_text


def create_embeddings_from_text(text_data: str, doc_id: str):
    """
    :param text_data:
    :param doc_id:
    :return:
    """
    # if len(text_file_path):
    #     with open(text_file_path, "r") as file:
    #         text_data = file.read()
    chunks = split_into_chunks(text_data)
    embeddings_path = create_embedding(chunks, doc_id, save_df=True, debug=DEBUG)
    return embeddings_path


def num_tokens(text: str, model: str = ANSWERING_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(
            f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens."
        )
    return truncated_string


def split_strings_into_chunks(
    text_string: str,
    max_tokens: int = 1000,
    model: str = ANSWERING_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    num_tokens_in_string = num_tokens(text_string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [text_string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(text_string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text_string, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_strings = split_strings_into_chunks(
                        half,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(text_string, model=model, max_tokens=max_tokens)]


def split_into_chunks(file_contents):
    # split sections into chunks
    chunks = []
    subsections = file_contents.split("\n\n\n")
    for section in subsections:
        chunks.extend(split_strings_into_chunks(section, max_tokens=MAX_TOKENS))

    print(f"{len(subsections)} sections split into {len(chunks)} strings.")
    return chunks


def create_embedding(chunks, doc_id, save_df=False, debug=False):
    embeddings = []
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = chunks[batch_start:batch_end]
        print(f"Batch {batch_start} to {min(len(chunks), batch_end - 1)}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert (
                i == be["index"]
            )  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": chunks, "embedding": embeddings})
    save_path = EMBEDDINGS_SAVE_PATH
    if save_df:
        # write to mongodb
        hotel_emb_obj = {"doc_id": doc_id, "text_chunks": []}
        for i in df.index:
            chunk = dict()
            chunk["chunk_id"] = i
            chunk["text"] = df.loc[i, "text"]
            chunk["embedding"] = df.loc[i, "embedding"]
            hotel_emb_obj["text_chunks"].append(chunk)
        mongo_collection.insert_one(hotel_emb_obj)
        print("Successfully saved embeddings to mongodb")

        if debug:
            df.to_csv(save_path, index=False)
            print(f"CSV file saved to {save_path}")

    return save_path
