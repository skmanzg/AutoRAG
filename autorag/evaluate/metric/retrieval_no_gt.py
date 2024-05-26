from typing import List, Dict, Optional

import pandas as pd

from autorag.utils.util import make_generator_callable_params


def retrieval_precision_tonic(queries: List[str], retrieved_contents: List[List[str]],
                              generator_module: Optional[Dict] = None) -> List[float]:
    """
    Calculate retrieval precision from tonic validate.
    It uses LLM to calculate retrieval precision score.
    It needs intensive computation of LLM, so you must be careful when using it.
    Especially, DO NOT RECOMMEND TO USE GPT-4 or CLAUDE-3 OPUS on this metric.

    :param queries: A list of queries.
    :param retrieved_contents: A list of retrieved contents.
    :param generator_module: The dict of generator module.
        As default, it uses gpt-3.5-turbo openai_llm.
    :return:
    """
    if generator_module is None:
        generator_module = {
            'module_type': 'openai_llm',
            'llm': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 16,
        }

    generator_modules, generator_params = make_generator_callable_params({'generator_modules': [generator_module]})
    generator_func = generator_modules[0].__wrapped__
    generator_param = generator_params[0]

    # prompt from tonic validate library
    prompt_format = ("Considering the following question and context, determine whether the context "
                     "is relevant for answering the question. If the context is relevant for "
                     "answering the question, respond with true. If the context is not relevant for "
                     "answering the question, respond with false. Respond with either true or false "
                     "and no additional text.")
    df = pd.DataFrame({'query': queries, 'retrieved_content': retrieved_contents})
    exploded_df = df.explode(['retrieved_content'])
    exploded_df['prompt'] = exploded_df.apply(lambda row: f"{prompt_format}\nQUESTION: {row['query']}"
                                                          f"\nCONTEXT: {row['retrieved_content']}", axis=1)

    llm_generated_answers, _, _ = generator_func(prompts=exploded_df['prompt'].tolist(),
                                                 **generator_param)
    exploded_df['generated_answer'] = llm_generated_answers
    exploded_df['parsed_result'] = exploded_df['generated_answer'].apply(parse_boolean_response)

    df['result'] = exploded_df.groupby(level=0, sort=False)['parsed_result'].apply(list).tolist()
    df['metric_result'] = df['result'].apply(calculate_metric)
    return df['metric_result'].tolist()


def calculate_metric(result: List[bool]) -> float:
    return sum(result) / len(result)


def parse_boolean_response(response: str) -> bool:
    """Code from tonic validate package.

    Parse boolean response from LLM evaluator.

    Attempts to parse response as true or false.

    Parameters
    ----------
    response: str
        Response from LLM evaluator.

    Returns
    -------
    bool
        Whether response should be interpreted as true or false.
    """
    response_lower = response.lower()
    if response_lower == "true":
        return True
    if response_lower == "false":
        return False
    if "true" in response_lower and "false" not in response_lower:
        return True
    if "false" in response_lower and "true" not in response_lower:
        return False
    raise ValueError(
        f"Could not determine true or false from response {response_lower}"
    )
