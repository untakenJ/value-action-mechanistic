# Behavioral steering 实验

本实验在原有 baseline 的生成、checkpoint、LLM Judge 和结果导出链路上增加两种
steering。一次 run 表示一个固定条件；单因子和多因子都通过逗号分隔的
`--steer-factors` 配置。Judge 仍只看到原 behavioral prompt 与模型回答，不看到实验
条件元数据，避免把目标因子名称直接泄露给 Judge。

## Prompt steering

该方法在最后一条 user message 末尾追加以下句子：

| 因子 | 追加句子 |
|---|---|
| Responsiveness | `Make the answer responsive.` |
| Deference | `Make the answer deferential.` |
| Boldness | `Make the answer bold.` |
| Guardedness | `Make the answer guarded.` |
| Verbosity | `Make the answer verbose.` |

单因子：

~~~bash
uv run python -m psycho_llm_behavioral \
  --steer-method prompt \
  --steer-factors Responsiveness
~~~

多因子会按给定顺序追加多个句子：

~~~bash
uv run python -m psycho_llm_behavioral \
  --steer-method prompt \
  --steer-factors Responsiveness,Boldness,Verbosity
~~~

Prompt steering 同时支持 `hf` 和 `vllm` 后端。

## J-lens Injected-thought steering

实现遵循 J-lens 论文的写入定义。对每个目标 token `t`，在配置层 `l` 取

~~~text
v_t = row_t(W_U J_l)
~~~

并在最后一个 user turn 的每个内容 token 位置执行

~~~text
h_l <- h_l + alpha * sum_t(v_t)
~~~

向量不做单位范数归一化；因此 `alpha` 是论文公式中的原始、未归一化系数。干预只在
prompt prefill 发生，不作用于随后生成的 Assistant token。受干预 forward 产生的 KV
cache 会被同一个推理后端直接用于后续 decode。

使用仓库默认 Gemma 3 4B lens 和 vLLM 连续批处理：

~~~bash
uv run python -m psycho_llm_behavioral \
  --backend vllm \
  --steer-method jlens \
  --steer-factors Responsiveness,Boldness \
  --steer-layer 20 \
  --steer-alpha 1.0
~~~

Hugging Face hook 路径可用于对照或后续适配其他架构：

~~~bash
uv run python -m psycho_llm_behavioral \
  --backend hf \
  --batch-size 1 \
  --steer-method jlens \
  --steer-factors Guardedness \
  --steer-layer 20 \
  --steer-alpha 1.0
~~~

主要参数：

| 参数 | 含义 |
|---|---|
| `--steer-factors` | 一个或多个因子名称；也接受 RE/DE/BO/GU/VB 或配置中的 `prompt_adjective` |
| `--steer-layer` | 从 0 开始的 transformer block 输出层；必须存在于 fitted lens |
| `--steer-alpha` | 应用于所有已解析 token 方向之和的全局未归一化系数 |
| `--steer-concept-config` | 本次实验使用的 JSON 概念配置；默认见 [concept_configs/default.json](concept_configs/default.json) |
| `--jlens-repo` | Hugging Face lens repo，或本地 checkpoint/目录 |
| `--jlens-file` | repo/目录内的 `.pt` 文件路径 |
| `--steer-token FACTOR=TOKEN` | 兼容旧命令：用一个精确 token 或 `id:INTEGER` 替换该因子的全部配置项目 |

### 概念配置

默认配置位于 [concept_configs/default.json](concept_configs/default.json)，保持原实现的
`responsive / deferential / bold / guarded / verbose` 五个概念。每个因子包含：

- `prompt_adjective`：供 prompt steering 和因子形容词别名使用。
- `concepts`：J-lens 使用的有序项目列表；项目不会去重，所有解析出的 token 方向直接相加。

字符串项目是 `{"text": "..."}` 的简写。文本项目支持：

- `prepend_space`：默认 `true`，编码前添加一个空格；设为 `false` 可选择词首版本。
- `token_selection`：默认 `"first"`，只取首 token；设为 `"all"` 时使用完整分词中的每个 token。
- `label`：可选审计标签，不影响 tokenization。

也可以使用 `{"token_id": 16627}` 直接指定 vocabulary row。Token ID 绑定 tokenizer；
代码会在解析时发出警告，更换模型或 tokenizer 后必须重新验证。以下配置会把带空格和
不带空格的 `bold`、`audacious` 的全部 fragment，以及一个直接 ID 作为四个独立项目
叠加；即使两个项目最终解析到同一 ID，也仍会重复相加：

~~~json
{
  "schema_version": 1,
  "name": "bold-multi",
  "factors": {
    "BO": {
      "prompt_adjective": "bold",
      "concepts": [
        {"text": "bold"},
        {"text": "bold", "prepend_space": false},
        {"text": "audacious", "token_selection": "all"},
        {"token_id": 16627, "label": "Gemma-3 bold row"}
      ]
    }
  }
}
~~~

运行时显式选择配置：

~~~bash
uv run python -m psycho_llm_behavioral \
  --backend vllm \
  --steer-method jlens \
  --steer-factors Boldness \
  --steer-concept-config configs/bold-multi.json \
  --steer-layer 20 \
  --steer-alpha 1.0
~~~

配置文件可以只定义本次需要的因子；若 `--steer-factors` 选择了配置中不存在的因子，
启动时会报错。配置内容、绝对来源路径和规范化内容 SHA-256 会写入实验元数据和 generation
fingerprint。每个解析结果另行记录 `concept_item_index`、完整 `concept_token_ids`、
实际 `token_id/token_text` 和所选 fragment 下标。

旧的 `--steer-token 'Deference= defer'` 与 `Deference=id:47634` 仍然可用，但其语义是
用一个 token 替换该因子在配置中的全部 `concepts`；多项目实验应使用 JSON 配置。

### vLLM 当前支持边界

- 当前运行时 hook 路径针对 Gemma 3，tensor parallel 固定为 1。
- hook 通过本地 `LLM.apply_model` 安装和审计；vLLM V1 因而需要启用
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`（backend 会自动设置）。只应在可信的本地代码、
  模型和 lens checkpoint 环境中运行。
- 为保证每个 prefill 都能解析完整最后 user turn，关闭 chunked prefill 和 prefix caching。
- worker 使用 eager mode 以保留层 hook；仍使用 vLLM 的高效 kernels、KV cache 与连续批处理。
- vLLM 路径要求 chat template；`--raw-prompt` 请改用 HF 路径。
- HF 路径不绑定 Gemma 类，但模型 hidden size、tokenizer 和 fitted lens 必须匹配。

## 输出与 resume

非 baseline 条件的默认目录名包含方法、因子、层、alpha、配置名和配置内容哈希，例如：

~~~text
outputs/psycho_llm_behavioral/google__gemma-3-4b-it__jlens__DE-VB__L20__a1.5__cfg-default-a893e5f2/
~~~

`manifest.json` 和每条 `model_responses.jsonl` 都保存 steering 配置；J-lens 响应另外保存
解析后的 token、prompt token 数、实际注入位置及 worker 批级注入计数审计。`results.csv` 增加 steering 条件列，
Judge 的五因子原始分和校正分与 baseline 完全一致。Baseline 的既有 fingerprint 与
response ID 保持不变。

## 建议的实验设计

不要只报告一个任意的层和 alpha。建议至少包含：

1. 同模型、同 sampling seed 的 baseline、prompt steering 和 J-lens steering。
2. 在中间 workspace 层附近做 layer sweep，并包含相邻层作为稳健性检查。
3. 对 alpha 做弱到强的 sweep；也可加入负值作为反向干预，并监控输出退化、重复和困惑度代理指标。
4. 单因子条件与预先声明的多因子组合分开报告；多因子使用向量和，效应不必线性相加。
5. 为多 token 因子比较默认首 fragment 与人工选择的单 token 同义词，并在结果中报告
   实际 token ID，而不是只报告自然语言标签。

目前还值得后续开放配置的内容包括：逐因子 alpha、多个注入层、注入位置子集、向量
归一化策略、随机方向/零向量 ablation，以及自动 layer × alpha sweep。它们会改变干预定义，
建议作为显式实验条件加入，而不要静默改变默认实现。

J-lens 方法来源：<https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens>
