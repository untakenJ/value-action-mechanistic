# 开放行为题复现

本目录复现 Contreras (2026) 的开放行为题部分：

- 20 个行为提示，每个因子 4 题。
- 默认每题采样 5 次，共 100 条模型原始输出。
- 生成参数与论文一致：temperature 1.0、最大输出 2048 tokens、无 system prompt。
- 两道 Responsiveness 题保留作者提供的三消息多轮上下文。
- Judge 对每条输出分别给出 RE、DE、BO、GU、VB 五个 1-5 分。
- 每个因子独立使用可复现的正向/反向表述，随后统一反向校正。
- Judge prompt 包含作者仓库中的 4 个合成 few-shot 校准例。

论文使用 Claude Opus 4.6、GPT-5.4、Gemini 3.1 Pro 三模型 Judge 集成。本复现按项目需求默认使用单个 DeepSeek V4 Pro，因此不应把单 Judge 结果表述为论文三模型集成结果。

## 环境

项目统一使用现有 uv 环境：

~~~bash
uv sync
uv sync --extra vllm
~~~

在项目根目录的 .env 中至少配置：

~~~dotenv
HF_TOKEN=hf_...
DEEPSEEK_API_KEY=sk-...
~~~

Gemma 3 是 gated model，Hugging Face 账号还需要接受对应模型许可。所有可配置项见根目录的 .env.example。

## 运行

先检查计划，不加载模型或调用 Judge：

~~~bash
uv run python -m psycho_llm_behavioral --dry-run
~~~

使用默认配置执行完整实验：

~~~bash
uv run python -m psycho_llm_behavioral
~~~

默认本地配置：

| 配置 | 默认值 |
|---|---|
| 被测模型 | google/gemma-3-4b-it |
| 后端 | vLLM |
| dtype | bfloat16 |
| vLLM batch size | 64 |
| vLLM GPU memory utilization | 0.85 |
| Judge | deepseek-v4-pro |
| Judge thinking | disabled |
| Judge workers | 4 |

Hugging Face 后端沿用 value-action-gap 的顺序生成方式，并启用 bf16 与 SDPA：

~~~bash
uv run python -m psycho_llm_behavioral --backend hf --batch-size 1
~~~

最小端到端测试：

~~~bash
uv run python -m psycho_llm_behavioral \
  --n-prompts 1 \
  --n-runs 1 \
  --judge-workers 1
~~~

以后需要增加采样时，在相同 run-name 下提高参数即可；默认 resume 会保留已有 run 1，只补跑新增编号：

~~~bash
uv run python -m psycho_llm_behavioral --n-runs 5
~~~

分阶段执行：

~~~bash
uv run python -m psycho_llm_behavioral --stages generate
uv run python -m psycho_llm_behavioral --stages judge
~~~

更换被测模型：

~~~bash
uv run python -m psycho_llm_behavioral --model org/model-name
~~~

更换 Judge 时可设置 JUDGE_BASE_URL、JUDGE_MODEL 和 JUDGE_API_KEY。若服务不接受 DeepSeek 的 thinking 扩展字段，设置 JUDGE_THINKING=omit。

两种 steering、单/多因子、J-lens 层与 alpha 配置见 [STEERING.md](STEERING.md)。

### 检查生成前最后一个 token 的 J-space

独立脚本会先用与行为实验相同的多轮 messages 和 chat template 渲染输入，然后固定检查
位置 `-1`，也就是模型生成第一个输出 token 之前的最后一个输入 token：

~~~bash
uv run python -m psycho_llm_behavioral.inspect_jspace \
  --prompt-ids BO-BP01,DE-BP03 \
  --layers all \
  --jspace-layers 65% \
  --top-k 10 \
  --jspace-k 25
~~~

输出同时保留两类不能混为一谈的结果：

- `top_lens_tokens`：论文的标准 Jacobian-lens readout，即对
  `W_U norm(J_l h_l)` 的 pre-softmax logits 排序。
- `jspace_decomposition`：用 token 对应的 `W_U J_l` 行向量做稀疏非负
  gradient pursuit，给出坐标系数以及 J-space/剩余部分的平方范数占比。

默认对所有已拟合层做标准 readout，但只在最接近模型 65% 深度的已拟合层做计算更昂贵的
稀疏分解；`--jspace-layers` 也可指定逗号分隔的实际层号，`--no-sparse-decomposition`
可只跑标准 readout。脚本不会静默截断超长输入，因为截断会改变“生成前最后一个 token”的
含义。默认 JSONL 写入
`outputs/psycho_llm_behavioral/jspace_last_token.jsonl`，每条记录包含完整渲染输入、被检查
token 的 id/文本/邻近上下文、各层结果，以及模型和 lens 元数据。

## 输出

默认输出目录为：

~~~text
outputs/psycho_llm_behavioral/google__gemma-3-4b-it/
~~~

| 文件 | 内容 |
|---|---|
| manifest.json | 论文来源、模型、后端、采样参数和 Judge 配置，不含密钥 |
| model_responses.jsonl | 完整输入消息、steering/注入元数据、模型原始输出、状态、时间和输出哈希 |
| judge_ratings.jsonl | Judge 原始 JSON、正反向 keying、原始分和反向校正后的五因子分 |
| results.csv | 模型输出、steering 条件与 Judge 分数的平面连接表 |
| summary.json | 总体、逐题和逐目标因子的均值与样本数 |

写入采用原子 upsert。默认开启 resume，只跳过成功且仍绑定当前模型输出哈希的记录；失败调用和过期评分会在重跑时更新。

五个校正后分数均为“越高表示该因子越强”：

| 代码 | 因子 |
|---|---|
| RE | Responsiveness |
| DE | Deference |
| BO | Boldness |
| GU | Guardedness |
| VB | Verbosity |

## 验证

测试不加载真实模型，也不访问外部 API：

~~~bash
uv run python -m unittest discover -v
~~~

## 来源

- Paper: https://arxiv.org/abs/2606.09843
- Author repository: https://github.com/jm-contreras/psycho-llm
- J-lens paper: https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens
- 行为提示材料：CC-BY 4.0
- 作者 pipeline 代码：MIT

使用这些提示或量表时，应引用原论文。
