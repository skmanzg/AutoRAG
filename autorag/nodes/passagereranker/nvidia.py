import asyncio
import os
from typing import List, Tuple, Optional

import aiohttp

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import process_batch

NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


@passage_reranker_node
def nvidia_reranker(queries: List[str], contents_list: List[List[str]],
                    scores_list: List[List[float]], ids_list: List[List[str]],
                    top_k: int, api_key: Optional[str] = None,
                    model: str = "nv-rerank-qa-mistral-4b:1",
                    batch: int = 8
                    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY", None)
        if api_key is None:
            raise ValueError("API key is not provided."
                             "You can set it as an argument or as an environment variable 'NVIDIA_API_KEY'")

    tasks = [nvidia_reranker_pure(query, contents, scores, ids, top_k=top_k, api_key=api_key, model=model) for
             query, contents, scores, ids in
             zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch))

    content_result, id_result, score_result = zip(*results)

    return list(content_result), list(id_result), list(score_result)


async def nvidia_reranker_pure(query: str, contents: List[str],
                               scores: List[float], ids: List[str],
                               top_k: int, api_key: str,
                               model: str = "nv-rerank-qa-mistral-4b:1") -> Tuple[List[str], List[str], List[float]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "query": {
            "text": query
        },
        "passages": [
            {
                "text": content
            }
            for content in contents
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(NVIDIA_API_URL, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            resp_json = await resp.json()

            if 'rankings' not in resp_json:
                raise RuntimeError(f"Invalid response from NVIDIA API: {resp_json}")

            results = resp_json['rankings'][:top_k]
            indices = [result['index'] for result in results]
            score_result = [result['logit'] for result in results]
            id_result = [ids[index] for index in indices]
            content_result = [contents[index] for index in indices]

    return content_result, id_result, score_result
