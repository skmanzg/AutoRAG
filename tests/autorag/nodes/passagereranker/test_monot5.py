from unittest.mock import patch

import pytest
import torch
from torch import FloatTensor
from transformers import T5ForConditionalGeneration
from transformers.generation import GenerateEncoderDecoderOutput

from autorag.nodes.passagereranker import monot5
from tests.autorag.nodes.passagereranker.test_passage_reranker_base \
    import (base_reranker_test, base_reranker_node_test, queries_example,
            contents_example, scores_example, ids_example, project_dir, previous_result)
from tests.delete_tests import is_github_action


def mock_t5_model_generate(input_ids, **kwargs) -> GenerateEncoderDecoderOutput:
    input_num = input_ids.size()[0]
    tokenizer_size = 32128
    first_score = FloatTensor(torch.zeros((6, tokenizer_size)))
    second_score = []
    for _ in range(input_num):
        tensor = torch.randn(1, tokenizer_size)
        second_score.append(tensor)

    second_score = FloatTensor(torch.cat(second_score, dim=0))

    return GenerateEncoderDecoderOutput(
        scores=(first_score, second_score),
    )


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
@patch.object(T5ForConditionalGeneration, 'generate', mock_t5_model_generate)
def test_monot5():
    top_k = 1
    original_monot5 = monot5.__wrapped__
    contents_result, id_result, score_result \
        = original_monot5(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5_batch_one():
    top_k = 1
    batch = 1
    original_monot5 = monot5.__wrapped__
    contents_result, id_result, score_result \
        = original_monot5(queries_example, contents_example, scores_example, ids_example, top_k, batch=batch)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5_node():
    top_k = 1
    result_df = monot5(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
