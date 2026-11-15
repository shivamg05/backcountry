# Real text inputs

`text.jsonl.gz` holds the texts that prompts for language-model jobs are cut
from: one JSON line per text, `{"kind", "text"}`. `tools/build_text_pool.py`
rebuilds it. Prompts are made by the harness (`src/backcountry/harness/workload.py`); they
are inputs to time and check the model on, never shown or published.

| kind | texts | source | license |
|---|---|---|---|
| chat | 85 conversations, 11 languages | [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2), validation split | Apache-2.0 |
| web | 100 pages | [C4](https://huggingface.co/datasets/allenai/c4) (English), validation split | ODC-BY 1.0; also subject to the Common Crawl terms of use |
| book | 24 excerpts of 12 books | [PG-19](https://huggingface.co/datasets/deepmind/pg19) (via the emozilla/pg19 Parquet copy), validation split; public-domain Project Gutenberg books | Apache-2.0 |
| math | 100 problems with worked answers | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), test split | MIT |
| code | 38 files | MLX 0.32.4 C++ headers and mlx-lm 0.31.3 Python | MIT, notice below |

## MLX and mlx-lm notice

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
