import asyncio
from unittest.mock import patch

from aioresponses import aioresponses

import autorag
from autorag.nodes.passagereranker import nvidia_reranker
from autorag.nodes.passagereranker.nvidia import nvidia_reranker_pure, NVIDIA_API_URL
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test


def test_nvidia_reranker_pure():
    with aioresponses() as m:
        mock_response = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }
        m.post(NVIDIA_API_URL, payload=mock_response)
        content_result, id_result, score_result = asyncio.run(
            nvidia_reranker_pure(queries_example[0], contents_example[0], scores_example[0], ids_example[0], top_k=2,
                                 api_key="mock_api_key"))
        assert len(content_result) == 2
        assert len(id_result) == 2
        assert len(score_result) == 2

        assert all([res in contents_example[0] for res in content_result])
        assert all([res in ids_example[0] for res in id_result])

        # check if the scores are sorted
        assert score_result[0] >= score_result[1]


async def mock_nvidia_reranker_pure(query, contents, scores, ids, top_k, api_key, **kwargs):
    if query == queries_example[0]:
        return [contents[1], contents[2], contents[0]][:top_k], [ids[1], ids[2], ids[0]][:top_k], [0.8, 0.2, 0.1][
                                                                                                  :top_k]
    elif query == queries_example[1]:
        return [contents[2], contents[0], contents[1]][:top_k], [ids[2], ids[0], ids[1]][:top_k], [0.8, 0.2, 0.1][
                                                                                                  :top_k]
    else:
        raise ValueError(f"Unexpected query: {query}")


# @patch.object(autorag.nodes.passagereranker.nvidia, "nvidia_reranker_pure", mock_nvidia_reranker_pure)
def test_nvidia_reranker():
    top_k = 3
    original_nvidia_reranker = nvidia_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_nvidia_reranker(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(autorag.nodes.passagereranker.nvidia, "nvidia_reranker_pure", mock_nvidia_reranker_pure)
def test_nvidia_reranker_batch_one():
    top_k = 3
    batch = 1
    original_nvidia_reranker = nvidia_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_nvidia_reranker(queries_example, contents_example, scores_example, ids_example, top_k, batch=batch)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(autorag.nodes.passagereranker.nvidia, "nvidia_reranker_pure", mock_nvidia_reranker_pure)
def test_nvidia_reranker_node():
    top_k = 1
    result_df = nvidia_reranker(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
