"""Offline controller and a bounded OpenAI Responses function-calling adapter."""
import json
import os
from urllib.request import Request, urlopen
from src.agent.prompts import SYSTEM_PROMPT


class MockController:
    """Deterministic workflow for integration tests; never presented as an LLM agent."""
    metadata = {'kind': 'mock', 'model': None}

    def choose(self, state, history, schemas):
        for name in ('get_predictions', 'get_company_context', 'get_previous_holdings'):
            if name not in state['reads']:
                return {'name': name, 'arguments': {}}
        if state['checked']:
            return {'name': 'write_portfolio_report', 'arguments': {}}
        if state['has_weights'] and not state['last_error']:
            return {'name': 'check_constraints', 'arguments': {}}
        if not state['has_candidates'] or state['last_error']:
            unused = [p for p in state['approved_plans'] if p['name'] not in state['used_plans']]
            if not unused:
                raise RuntimeError('no unused approved candidate plan')
            return {'name': 'rank_candidates', 'arguments': {'plan': unused[0]['name']}}
        return {'name': 'optimize_weights', 'arguments': {}}


class OpenAIController:
    """No SDK required. One stateless Responses request per step; bounded timeout.

    Sends anonymized observations only. No browsing/files/tools beyond supplied
    function schemas. Requires explicit model configuration and OPENAI_API_KEY.
    transport is injectable for offline contract tests. No HTTP retries hidden
    inside the controller; failures stop the run, avoiding unbounded spending.
    """
    def __init__(self, model, *, timeout=45, max_output_tokens=1200, transport=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError('explicit OpenAI model required')
        if not 0 < timeout <= 60 or type(max_output_tokens) is not int or not 256 <= max_output_tokens <= 8000:
            raise ValueError('invalid API timeout or output token limit')
        self.model, self.timeout, self.max_output_tokens = model, timeout, max_output_tokens
        self.transport = transport or self._request
        self.key = os.environ.get('OPENAI_API_KEY')
        if transport is None and not self.key:
            raise ValueError('OPENAI_API_KEY is not configured; offline smoke remains available')
        self.metadata = {'kind': 'openai_responses', 'model': model, 'timeout': timeout,
                         'max_output_tokens': max_output_tokens, 'store': False}
        self.calls = []

    def _request(self, payload):
        request = Request('https://api.openai.com/v1/responses',
                          data=json.dumps(payload, allow_nan=False).encode(), method='POST',
                          headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def choose(self, state, history, schemas):
        payload = {'model': self.model, 'instructions': SYSTEM_PROMPT,
                   'input': [{'role': 'user', 'content': json.dumps({'state': state, 'history': history}, allow_nan=False)}],
                   'tools': schemas, 'tool_choice': 'required', 'parallel_tool_calls': False,
                   'max_output_tokens': self.max_output_tokens, 'store': False}
        response = self.transport(payload)
        self.calls.append({'id': response.get('id'), 'model': response.get('model'), 'usage': response.get('usage')})
        if response.get('status') != 'completed':
            raise RuntimeError('Responses request incomplete or failed')
        calls = [x for x in response.get('output', []) if x.get('type') == 'function_call']
        if len(calls) != 1:
            raise ValueError('controller must return exactly one function call')
        call = calls[0]
        args = json.loads(call['arguments'])
        if type(args) is not dict:
            raise ValueError('function arguments must be an object')
        return {'name': call['name'], 'arguments': args}
