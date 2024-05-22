from autorag.nodes.passagefilter import rse
from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, \
    scores_example, ids_example, base_passage_filter_test


def test_rse():
    original_rse = rse.__wrapped__
    contents, ids, scores = original_rse(
        queries_example, contents_example, scores_example, ids_example
    )
    base_passage_filter_test(contents, ids, scores)
