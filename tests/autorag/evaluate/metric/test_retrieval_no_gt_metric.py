import time
from unittest.mock import patch

import openai.resources.chat
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionTokenLogprob
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs

from autorag.evaluate.metric import retrieval_precision_tonic

query = ['What is the capital of France?', 'How many members are in Newjeans?']
retrieved_contents = [
    ['Paris is the capital of France.', 'Paris is one of the capital from France. Isn\'t it?'],
    ['Newjeans has 5 members.', 'Danielle is one of the members of Newjeans.']
]


async def mock_openai(self, messages, model, **kwargs):
    return ChatCompletion(
        id='test_id',
        choices=[Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content="true",
                role='assistant',
            ),
            logprobs=ChoiceLogprobs(
                content=[
                    ChatCompletionTokenLogprob(
                        token='true',
                        logprob=-0.445,
                        top_logprobs=[],
                    ),
                ]
            ),
        )],
        created=int(time.time()),
        model=model,
        object='chat.completion',
    )


@patch.object(openai.resources.chat.completions.AsyncCompletions, 'create', mock_openai)
def test_retrieval_precision_tonic():
    result = retrieval_precision_tonic(query, retrieved_contents)
    assert isinstance(result, list)
    assert isinstance(result[0], float)
    assert all(res == 1.0 for res in result)
