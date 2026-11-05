# 多上游接入与渠道级协议兼容

目标是复用 Done Hub 的已有 provider，使客户端保持原入口，按**渠道公开的协议**接入官方或第三方中转平台。不能从模型名、响应 ID 或“平台背后用了 AWS”推定其认证/协议，也不承诺未知平台零配置兼容。

## 1. 已有适配器：优先复用，不按转售商新建类型

以下是当前源码能力，不等于每个账号/最新模型都已完成现场验收：

| 对我们公开的接口/认证 | 现有渠道类型 | 已有主链与限制 |
|---|---:|---|
| OpenAI Chat / Responses + Bearer Key | 1 OpenAI | 标准 Chat、原生 Responses；可按模型开启下面的 Chat→Responses 桥接 |
| 自定义 OpenAI 兼容路径 | 8 自定义 | 复用 OpenAI provider；`plugin.customize` 已可配置 Chat/Responses 等地址，不是任意协议翻译器 |
| Anthropic Messages + API Key | 14 Anthropic | Chat→Messages→Chat，以及 `/claude/v1/messages` 原生路径；最新专属字段推荐原生入口 |
| Gemini generateContent + API Key/服务账号 | 25 Gemini | Chat→Gemini→Chat 与原生 `/gemini/...`；已有模型/参数/签名处理。缺签名降级必须明确启用 |
| AWS Bedrock InvokeModel / Responses | 32 Bedrock | Region+IAM/SigV4 或 Bearer；Chat 按模型族转换、支持 Claude 原生入口；Responses 按模型能力判断，不是所有模型通用 |
| Google Vertex AI | 42 Vertex AI | 服务账号、Project/Region、原生/Chat 转换已有；不等于复用 type-25 新增的签名补救 |
| Gemini CLI | 57 Gemini CLI | 已有 OAuth、Chat/Gemini 调用适配；需账号与真实请求验收，不是 API Key 渠道 |
| Claude Code | 58 Claude Code | 已有 OAuth/刷新与 CLI 请求适配、Chat 转换；当前未实现原生 `ClaudeChatInterface`，不能只因出现在类型列表就宣称所有入口可用 |
| Codex OAuth | 59 Codex | 已有 Chat/Responses 适配；其 Chat 工具流仍须独立验收，不应假设与 type-1 共享全部转换行为 |
| Vertex AI Express | 61 Vertex Express | 已有 Key/Chat/Gemini 路径，需单独验证能力边界 |
| AWS Bedrock Messages/Mantle | 62 Bedrock Messages | 已有 SigV4/Bearer + 原生 `/anthropic/v1/messages`；当前只实现原生 Messages，不实现 ChatInterface |

例如供应商通过 `https://platform.example/v1/chat/completions` 和 Bearer Key 提供 Claude，即使供应商后面连接 Bedrock，我们仍应按其公开契约选 OpenAI；不能把中转 Key 当 AWS IAM 凭据。

同一家供应商可按公开协议分成 GPT/Claude/Gemini 多个渠道；不同渠道可复用同一批准账号 Key，但公司/个人的账号、配置、数据和发布必须分别治理，不能因复用代码而同步个人实例。

## 2. GPT 工具需要 Responses：复用既有桥接

原代码已有 `ToResponsesRequest`、`relayChat.compatibleSend`、Responses→Chat JSON/SSE 转换，但只用固定模型表触发。现在为 type 1/8 在现有插件 UI 增加**模型级明确选择**，不增加类型编号、数据库表或第二代理：

```json
{
  "chat_to_responses": {
    "models": "gpt-6.1-sol,gpt-6-sol,gpt-6-astra"
  }
}
```

- 这是渠道 `plugin` 字段，不是 `custom_parameter`、Pi 配置或上游 JSON 参数。
- 模型列表按精确名称匹配（公开别名或映射后名称），逗号分隔；`*` 表示整个渠道，空/缺省关闭。
- 客户端继续发 `/v1/chat/completions`，本渠道改用 `/v1/responses`，最终返回 Chat 格式。
- 不根据域名/供应商/通用400猜协议，不悄悄替换模型。未支持的渠道类型/模型明确报 `unsupported_api`。
- 与既有 `compatible_response` **方向相反**：后者让 Responses 入口退回 Chat；不是本开关。
- type 8 的非标准路径继续使用已有 `plugin.customize["16"]`。
- JSON 与 SSE 均保留函数 call ID、并行工具索引、参数和结果；工具结束为 `tool_calls`，上游 failed/error 不伪装正常结束。最终 done 事件才返回参数时只补尚未下发的后缀，冲突/截断参数明确报错。
- Responses 桥接必须收到完成/失败/截断等协议终态；仅 HTTP EOF 或裸 `[DONE]` 不算成功，不再追加成功标记。非流式根/输出项 incomplete 的 `length` 语义优先于工具结束，避免执行半截调用。
- 原生 `/v1/responses` 默认不经此桥接，继续保留其已有字节透传能力。

边界：Chat 和 Responses 不是任意字段无损互转。当前主验收为文本、system/developer消息、图片URL、reasoning、函数工具及结果。对转换器无法表达的音频、文件、legacy functions、stop、n>1 等请求明确返回 `unsupported_conversion`，不丢字段后假成功。专属 hosted tools 等需求应按能力使用原生接口；仅列出模型不算验证。不能把更换 URL 当作协议转换。

## 3. 特殊参数：复用既有模型级自定义参数

例如供应商拒绝 temperature，而特定模型需要 reasoning：

```json
{
  "per_model": true,
  "gpt-6.1-sol": {
    "remove_params": ["temperature"],
    "reasoning": {"effort": "low"}
  },
  "another-model": {
    "remove_params": ["top_p"]
  }
}
```

这是渠道的 `custom_parameter`。处理最终上游 body，所以桥接后应使用 Responses 字段（如 `reasoning.effort` / `max_output_tokens`），Gemini 原生应使用 `generationConfig.*`。`overwrite=true` 才覆盖已有值；缺省只补缺失参数。`pre_add` 用于现有入口预处理，不要同时把同一规则当两套协议使用。

已有机制还包括模型映射、HeaderOverride、proxy、自定义端点与流式允许列表。优先使用这些机制，不为供应商某次422新增专属类型；也不应默认删除 tools/thinking/JSON Schema 约束来制造“成功”。当前 Claude/Gemini Chat 转原生处理的是已有 `reasoning` 对象，不能把 OpenAI `reasoning_effort` 宣称为所有 provider 都会自动映射；若供应商要求特殊 thinking 结构，应先按最终协议设置渠道参数并验收，不全局替模型开启 thinking。

## 4. Gemini 签名：有限、可见的协议补救

复用 type-25 的 `gemini_signature_autofill.enable`；首次保留真实签名，明确签名校验400后同渠道/同凭据最多重发一次，与首次共享既有超时预算。不开全局重试，不增长全局超时，不污染签名缓存，不改工具参数/结果；流已开始后报错则直接失败。详见 [Gemini compatibility](gemini-openai-compatibility.md)。

该方案对所有走这一 provider 的同协议上游生效，没有 apimodel 域名判断。哨兵可能丢失原思考状态，不宣称等价于真实签名；不验证接受哨兵的后端应保持开关关闭。

## 5. 新平台最短验收

1. 确认其公开 endpoint、认证、真实可调用模型，而非猜其内部供应商。
2. 选已有 provider；不同协议分渠道，保留公司独立配置/凭据。
3. 如需要，用现有模型映射、参数/头/路径配置；Responses-only 工具用明确桥接选择。
4. 用实际客户端跑文本→工具调用→回传→下一轮，含两个并行同名函数、SSE、失败与取消；仅 HTTP 200/健康/测速不算工具能力成功。
5. 保存 request ID、命中渠道、实际上游错误，区分路由失败、转换失败、账号无模型/额度、供应商容量。默认包装503不是“供应商饱和”的证据。

上游无模型、余额不足、限流或未给所需权限不能由本代码生成能力。某个代理支持适配器也不证明模型身份；必须明确记录已验/未验范围。
