from typing import List, Tuple

from sprag.rse import get_best_segments, get_meta_document, get_relevance_values

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def rse(queries: List[str], contents_list: List[List[str]],
        scores_list: List[List[float]], ids_list: List[List[str]],
        top_k_for_document_selection: int = 7,
        max_length: int = 10,
        overall_max_length: int = 50,
        minimum_value: float = 0.2,
        irrelevant_chunk_penalty: float = 0.2,
        decay_rate: int = 20,
        ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Apply the Relevance Score Extraction (RSE) algorithm to select the most relevant segments from the contents.

    :param queries: The list of queries to use for filtering
    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :param top_k_for_document_selection: The number of top documents to consider for each query
        Default is 7.
    :param max_length: The maximum length of a segment in chunks
        Default is 10.
    :param overall_max_length: The maximum total length of selected segments in chunks
        Default is 50.
    :param minimum_value: The minimum relevance value for a segment to be considered
        Default is 0.2.
    :param irrelevant_chunk_penalty: The penalty for irrelevant chunks in a segment
        Default is 0.2.
    :param decay_rate: The decay rate for the exponential decay function used to calculate relevance values
        Default is 20.
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    all_ranked_results = []
    for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list):
        ranked_results = [{"metadata": {"doc_id": doc_id, "chunk_index": chunk_index}, "score": score}
                          for doc_id, chunk_index, score in zip(ids, range(len(contents)), scores)]
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        all_ranked_results.append(ranked_results)

    document_splits, document_start_points, unique_document_ids = get_meta_document(all_ranked_results,
                                                                                    top_k_for_document_selection)
    meta_document_length = document_splits[-1] if document_splits else 0

    all_relevance_values = get_relevance_values(all_ranked_results, meta_document_length, document_start_points,
                                                unique_document_ids, irrelevant_chunk_penalty, decay_rate)

    best_segments = get_best_segments(all_relevance_values, document_splits, max_length, overall_max_length,
                                      minimum_value)

    filtered_contents_list = []
    filtered_ids_list = []
    filtered_scores_list = []
    for start, end in best_segments:
        for document_id in unique_document_ids:
            if start >= document_start_points[document_id] and end <= document_start_points[document_id] + \
                    document_splits[unique_document_ids.index(document_id)]:
                chunk_start = start - document_start_points[document_id]
                chunk_end = end - document_start_points[document_id]
                filtered_contents_list.append(contents_list[queries.index(query)][chunk_start:chunk_end])
                filtered_ids_list.append(ids_list[queries.index(query)][chunk_start:chunk_end])
                filtered_scores_list.append(scores_list[queries.index(query)][chunk_start:chunk_end])
                break

    return filtered_contents_list, filtered_ids_list, filtered_scores_list
