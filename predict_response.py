"""
Prediction methods
"""

# imports
import openai
import tiktoken
from scipy import spatial
from config import EMBEDDING_MODEL, ANSWERING_MODEL, TOP_K
from ai_compendium import mongo_collection


# search function
def strings_ranked_by_similarity(
    query: str,
    embedding_obj,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 1,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and similarity, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_similarities = [
        (item["text"], relatedness_fn(query_embedding, item["embedding"]))
        for item in embedding_obj[0]["text_chunks"]
    ]
    strings_and_similarities.sort(key=lambda x: x[1], reverse=True)
    strings, similarities = zip(*strings_and_similarities)
    return strings[:top_n], similarities[:top_n]


def num_tokens(text: str, model: str = ANSWERING_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query: str, embedding_obj, model: str, token_budget: int) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, similarity = strings_ranked_by_similarity(
        query, embedding_obj, top_n=TOP_K
    )
    introduction = (
        "You will be given some context and a question that follows that context."
        "Your job is to answer the question based on the context provided."
    )
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


def get_query_response(
    query: str,
    doc_id: str,
    db=mongo_collection,
    model: str = ANSWERING_MODEL,
    token_budget: int = 3200 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    embedding_obj = db.find({"doc_id": doc_id})
    message = query_message(
        query, embedding_obj, model=model, token_budget=token_budget
    )
    if print_message:
        print(message)
    messages = [
        {
            "role": "system",
            "content": "You answer the given question by referring to the context.",
        },
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=0
    )
    # print(f"OpenAI response: {response}")
    response_message = response["choices"][0]["message"]["content"]
    if response_message == "UNCERTAIN":
        response_message = "Sorry I do not have enough information to answer this. Please contact the reception."
    return response_message
