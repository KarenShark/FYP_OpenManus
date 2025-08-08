# OpenManus 阶段性开发指南

## 分支策略

为了更好地管理不同阶段的开发工作，我们采用分支策略来隔离各个 Stage 的开发：

### 分支结构

```
main                     # 主分支，保持稳定的代码
├── stage-1-*            # Stage 1 相关功能分支
├── stage-2-trading-tools # Stage 2 交易工具开发
├── stage-3-*            # Stage 3 相关功能分支（未来）
└── stage-4-*            # Stage 4 相关功能分支（未来）
```

### 当前分支状态

#### `main` 分支
- 保持原始的 OpenManus 代码库状态
- 包含核心 MCP 服务器和基础工具
- 所有新的 Stage 开发都从这里分出

#### `stage-2-trading-tools` 分支  ✅ 已完成
- **功能**: 交易平台工具层 & MCP 扩展
- **标签**: `v0.1.0-tools`
- **包含工具**:
  - `YahooFinanceFetcher`: 历史行情数据 (OHLCV)
  - `NewsAPIFetcher & EnhancedNewsFetcher`: 财经新闻 + 情感分析
  - `TechnicalIndicators`: SMA, EMA, RSI, MACD, Bollinger Bands
  - `PositionSizer`: 风险管理的头寸大小计算
  - `RiskAnalyzer`: 综合投资组合风险分析
- **测试**: 24个集成测试全部通过 ✅
- **状态**: 开发完成，可用于生产

## 开发工作流

### 开始新的 Stage 开发

1. **从 main 分支创建新分支**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b stage-X-feature-name
   ```

2. **进行开发工作**:
   - 实现新功能
   - 编写测试
   - 更新文档

3. **测试和验证**:
   ```bash
   # 运行测试
   pytest tests/ -v

   # 检查代码质量
   pre-commit run --all-files
   ```

4. **提交和标记**:
   ```bash
   git add .
   git commit -m "feat: implement stage-X functionality"
   git tag vX.X.X-stage-name
   ```

### 切换到不同的 Stage

```bash
# 查看所有分支
git branch -a

# 切换到 Stage 2 交易工具
git checkout stage-2-trading-tools

# 切换回主分支
git checkout main

# 切换到其他 Stage 分支
git checkout stage-X-feature-name
```

### 合并策略

每个 Stage 完成后，可以根据需要：

1. **保持分支独立**: 用于特定功能的演示和测试
2. **合并到主分支**: 当功能稳定且需要集成时
3. **创建发布分支**: 用于生产部署

## Stage 2 使用指南

### 安装和运行

```bash
# 切换到 Stage 2 分支
git checkout stage-2-trading-tools

# 安装依赖
pip install -r requirements.txt

# 运行交易工具测试
pytest tests/test_tool_integration.py -v

# 启动 MCP 服务器
python run_mcp_server.py
```

### 可用的交易工具

1. **历史数据获取**:
   ```python
   # 获取 AAPL 过去30天的数据
   yfinance_fetcher(symbol="AAPL")
   ```

2. **技术分析**:
   ```python
   # 计算技术指标
   technical_indicators(
       price_data=market_data,
       indicators=["SMA", "RSI", "MACD"]
   )
   ```

3. **风险管理**:
   ```python
   # 计算头寸大小
   position_sizer(
       account_balance=100000,
       risk_per_trade=0.02,
       entry_price=150.0,
       stop_loss_price=140.0
   )
   ```

4. **投资组合分析**:
   ```python
   # 分析投资组合风险
   risk_analyzer(
       positions=portfolio_positions,
       total_portfolio_value=100000
   )
   ```

### 工具特性

- ✅ **异步支持**: 所有工具支持异步执行
- ✅ **错误处理**: 完善的错误处理和验证
- ✅ **可配置**: 支持自定义参数
- ✅ **可扩展**: 易于添加新指标
- ✅ **生产就绪**: 包含日志和监控

## 未来规划

### Stage 3: 前端界面 (计划中)
- 交易分析仪表板
- 实时数据展示
- 用户交互界面

### Stage 4: 智能代理 (计划中)
- 自动交易建议
- 风险预警系统
- 智能投资组合优化

## 贡献指南

1. 为每个新 Stage 创建独立分支
2. 确保所有测试通过
3. 更新相关文档
4. 使用清晰的提交信息
5. 添加适当的标签

## 联系和支持

如有问题或建议，请在相应的分支创建 Issue 或 Pull Request。

---

**注意**: 此分支策略确保各个 Stage 的开发工作不会相互干扰，同时保持代码库的整洁和可维护性。
