From: https://chatgpt.com/share/68f8c9b5-8260-800e-b261-963a8171eb87

| Prefix              | Provider                                         | Notes                                                                                         |
| ------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `openai:`           | OpenAI Chat/Completion                           | Standard. ([Pydantic AI][1])                                                                  |
| `openai-responses:` | OpenAI “Responses API”                           | As documented. ([Pydantic AI][1])                                                             |
| `anthropic:`        | Anthropic                                        | e.g., Claude models. ([Pydantic AI][2])                                                       |
| `google:`           | Google LLC (Generative Language API / Vertex AI) | The source shows provider name can be `'google-vertex'` or `'google-gla'`. ([Pydantic AI][3]) |
| `bedrock:`          | Amazon Bedrock                                   | AWS Bedrock models. ([GitHub][4])                                                             |
| `cohere:`           | Cohere Inc.                                      | Cohere models. ([GitHub][4])                                                                  |
| `mistral:`          | Mistral AI                                       | Mistral models. ([GitHub][4])                                                                 |
| `grok:`             | Grok (from xAI)                                  | The docs list `grok:` prefix. ([Pydantic AI][2])                                              |
| `github:`           | GitHub Models                                    | Use `github:` prefix. ([Pydantic AI][2])                                                      |
| `vercel:`           | Vercel, Inc. AI Gateway                          | Docs show `vercel:` prefix. ([Pydantic AI][2])                                                |
| `ollama:`           | Ollama                                           | Self-hosted or cloud Ollama models. ([Pydantic AI][2])                                        |
| `azure:`            | Azure (OpenAI / AI Foundry)                      | Use `azure:` prefix. ([Pydantic AI][2])                                                       |
| `deepseek:`         | DeepSeek                                         | OpenAI-compatible API as separate provider. ([Pydantic AI][1])                                |
| `openrouter:`       | OpenRouter                                       | Use `openrouter:` prefix. ([Pydantic AI][1])                                                  |
| `together:`         | Together AI                                      | Use `together:` prefix. ([Pydantic AI][2])                                                    |
| `fireworks:`        | Fireworks AI                                     | Use `fireworks:` prefix. ([Pydantic AI][2])                                                   |
| `heroku:`           | Heroku                                           | Use `heroku:` prefix. ([Pydantic AI][2])                                                      |
| `cerebras:`         | Cerebras Systems                                 | Use `cerebras:` prefix. ([Pydantic AI][2])                                                    |
| `nebius:`           | Nebius AI Studio                                 | Use `nebius:` prefix. ([Pydantic AI][2])                                                      |
| `moonshotai:`       | MoonshotAI                                       | Use `moonshotai:` prefix. ([Pydantic AI][2])                                                  |
| `liteLLM:`          | LiteLLM (local/proxy)                            | Use `liteLLM:` prefix. ([Pydantic AI][2])                                                     |
| `custom/`           | Custom provider prefix                           | Indicated in docs for user-configured models. ([Pydantic AI][2])                              |

[1]: https://ai.pydantic.dev/models/openai/?utm_source=chatgpt.com "OpenAI - Pydantic AI"
[2]: https://ai.pydantic.dev/models/?utm_source=chatgpt.com "Model Providers - Pydantic AI"
[3]: https://ai.pydantic.dev/api/providers/?utm_source=chatgpt.com "pydantic_ai.providers - Pydantic AI"
[4]: https://github.com/pydantic/pydantic-ai?utm_source=chatgpt.com "Agent Framework / shim to use Pydantic with LLMs - GitHub"
