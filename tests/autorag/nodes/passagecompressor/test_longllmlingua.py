from typing import List

from autorag.nodes.passagecompressor import longllmlingua

queries = [
    "What is the capital of France?",
    "What is the meaning of life?",
]
retrieved_contents = [
    ["Paris is the capital of France.", "France is a country in Europe.", "France is a member of the EU."],
    ["The meaning of life is 42.", "The meaning of life is to be happy.", "The meaning of life is to be kind."],
]


def test_longllmlingua_default():
    result = longllmlingua.__wrapped__(queries, retrieved_contents, [], [])
    print(result)
    check_result(result)


def check_result(result: List[str]):
    assert len(result) == len(queries)
    for r in result:
        assert isinstance(r, str)
        assert len(r) > 0
        assert bool(r) is True
