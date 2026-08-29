# Licenses & 合规边界

> 原则：仓库内代码 Apache-2.0；**数据资产自带许可**，本文件只记录边界，不替代第三方原文。
> 发布（GitHub/public artifacts）前必须由本清单逐一确认（roadmap §5）。

## 1. 本项目（代码）

| 项 | 许可 | 备注 |
|---|---|---|
| EngramDB 源码（Rust/Python/脚本） | **Apache-2.0** | 见根 `LICENSE`；crates metadata `license = "Apache-2.0"` |
| 第三方 Rust/Python 依赖 | 各自 MIT/Apache-2.0/BSD | 依赖树内无 GPL/AGPL 强传染项（pandas 走 Apache-2.0？——以 `uv tree`/`cargo tree` 复核） |

## 2. 模型权重（用于开发与基准，不随仓库分发）

| 资产 | 许可/条款 | 边界（本文档结论） |
|---|---|---|
| Qwen3.8-Flash-Next 权重（33 safetensors，含 PLE 表） | **Qwen Community License 1.0** | 个人研究/开发 OK；提取 PLE 参数、嫁接实验、基准发布在**非商业渠道写明来源**；商业使用需单独授权。**model 权重的再发布视同模型再分发**，应避免直接托管，引用原始 repo 即可 |
| DeepSeek Engram demo（研究参照） | DeepSeek 代码仓库（Apache-2.0 风格）/论文 | 我们实现为 Rust 独立代码（按设计文档引用），仅算法参照；论文 arXiv 2601.07372 引用即可 |
| `transformers` qwen4_exp 实现 | Apache-2.0 | `refs/qwen4_exp_modeling.py` 仅作本地参考，未并入源码树发布 |

## 3. 语料与负载数据（统计产物：zipf/热集；不含正文）

| 数据源 | 许可/发布 | 我们在做的事 |
|---|---|---|
| FineWeb-Edu（tokens 统计） | *CC-BY-4.0（HuggingFace 目录说明）/ 学术使用* | 只用 token 行列统计（zipf/bigrams/rowid reach），**不发布原文**；发布统计表时注明来源 |
| Commons Crawl（cc_traces jsonl，已弃用为统计源） | 网页数据 CC-BY / crawl 自身条款 | 仅下载过，未纳入统计；如需引用以 crawl 许可为准 |
| `trace-commons`（agent 会话） | **CC-BY-4.0**（署名要求） | 统计口径同 FineWeb：发布数字时注明来源+署名 |
| `semianalysis`（agent 会话） | Apache-2.0（仓库声明） | 同上；注意 agent 数据可能包含对话隐私，**避免发布样例文本** |
| Gutenberg / 自有文本 | 各自公版/许可 | 本地开发用 |

**发布统计产物时的保守规则**：只公开“数值”与“生成图表”，不公开正文/会话原文；图表脚注统一加：`数据源: {FineWeb-Edu, trace-commons, semianalysis} (CC-BY-4.0 / Apache-2.0), statistics only`。

## 4. 待办（发布前置）

- [ ] `uv tree` / `cargo tree` 输出复核（含 build 依赖）定稿许可白名单
- [ ] github.com 发布：若公开仓库，附本文件 + README 消歧章节
- [ ] 商业咨询前：Qwen Community License 逐条核对（尤其"权重再分发"条款）
