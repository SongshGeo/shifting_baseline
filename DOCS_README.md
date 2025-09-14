# Past1000 文档系统使用说明

我已经为您的 Past1000 项目创建了完整的英文 API 文档系统，使用 MkDocs 和 Material 主题。

## 📁 文档结构

```
docs/
├── index.md                    # 项目主页
├── getting-started/            # 快速开始指南
│   ├── installation.md        # 安装指南
│   ├── quickstart.md          # 快速开始
│   └── configuration.md       # 配置指南
├── api/                       # API 参考文档
│   ├── data.md               # 数据模块
│   ├── compare.md            # 比较分析模块
│   ├── calibration.md        # 校准模块
│   ├── filters.md            # 过滤器模块
│   ├── abm.md                # 智能体模型模块
│   ├── process.md            # 处理模块
│   ├── constants.md          # 常量模块
│   └── utils.md              # 工具模块
├── examples/                  # 使用示例
│   ├── basic-usage.md        # 基础用法
│   ├── advanced-analysis.md  # 高级分析
│   └── abm-simulation.md     # ABM 模拟
└── development/               # 开发指南
    ├── contributing.md       # 贡献指南
    ├── testing.md           # 测试指南
    └── changelog.md         # 更新日志
```

## 🚀 快速开始

### 1. 安装文档依赖

```bash
pip install -r requirements-docs.txt
```

### 2. 本地预览文档

```bash
# 方法1：使用提供的脚本
python setup_docs.py serve

# 方法2：直接使用 mkdocs
mkdocs serve
```

文档将在 `http://localhost:8000` 打开。

### 3. 构建文档

```bash
# 方法1：使用提供的脚本
python setup_docs.py build

# 方法2：直接使用 mkdocs
mkdocs build
```

构建的文档将保存在 `site/` 目录中。

## 📚 文档特性

### Material 主题特性
- **响应式设计**：支持桌面、平板和手机
- **深色/浅色模式**：可切换主题
- **全文搜索**：搜索所有文档内容
- **导航系统**：分层导航，支持标签和章节
- **代码高亮**：语法高亮支持
- **提示框**：重要信息的提示框
- **标签页**：多代码示例的标签页
- **数学公式**：LaTeX 数学支持
- **图表**：Mermaid 图表支持

### MkDocs 插件
- **mkdocstrings**：自动生成 API 文档
- **search**：内置搜索功能
- **material**：Material Design 主题
- **pymdown-extensions**：额外的 markdown 扩展

## 🛠️ 配置说明

### MkDocs 配置 (mkdocs.yml)

主要配置包括：
- **主题**：Material 主题，支持深色/浅色模式
- **导航**：完整的导航结构
- **插件**：搜索、API 文档生成等
- **扩展**：数学公式、图表、代码高亮等

### 文档配置

- **API 文档**：自动从代码 docstring 生成
- **示例代码**：所有代码示例都经过测试
- **交叉引用**：模块间的交叉引用
- **搜索**：全文搜索支持

## 📝 文档内容

### 1. 项目概览
- 项目介绍和特性
- 快速开始示例
- 安装指南

### 2. API 参考
- **数据模块**：数据加载、处理和聚合
- **比较模块**：相关性分析和统计比较
- **校准模块**：不匹配分析和验证
- **过滤器模块**：数据过滤和分类
- **ABM 模块**：智能体建模框架
- **处理模块**：数据处理管道
- **常量模块**：常量和配置值
- **工具模块**：实用函数和帮助类

### 3. 使用示例
- **基础用法**：常见使用场景
- **高级分析**：复杂分析模式
- **ABM 模拟**：智能体建模示例

### 4. 开发指南
- **贡献指南**：如何参与项目开发
- **测试指南**：测试策略和最佳实践
- **更新日志**：版本历史和变更记录

## 🔧 自定义和扩展

### 添加新页面

1. 在相应目录创建新的 markdown 文件
2. 在 `mkdocs.yml` 中添加页面到导航
3. 按照现有风格编写内容
4. 使用 `mkdocs serve` 本地测试

### 更新 API 文档

1. 更新 Python 代码中的 docstring
2. 文档将自动更新
3. 本地测试更改

### 样式指南

- 使用清晰、简洁的语言
- 为所有函数包含示例
- 使用提示框标记重要信息
- 遵循现有的结构和格式
- 测试所有代码示例

## 🚀 部署

### GitHub Pages 部署

```bash
mkdocs gh-deploy
```

### 其他平台

构建的文档在 `site/` 目录中，可以部署到任何静态网站托管服务。

## 📋 文件说明

- `mkdocs.yml`：MkDocs 配置文件
- `requirements-docs.txt`：文档依赖
- `Makefile`：构建命令
- `setup_docs.py`：文档设置脚本
- `docs/`：文档源文件目录

## 🎯 下一步

1. **预览文档**：运行 `python setup_docs.py serve` 查看文档
2. **自定义内容**：根据需要修改文档内容
3. **添加示例**：添加更多使用示例
4. **部署**：部署到 GitHub Pages 或其他平台

## 📞 支持

如果您在使用文档系统时遇到问题：

1. 检查依赖是否正确安装
2. 查看 MkDocs 和 Material 主题文档
3. 检查 markdown 语法和 YAML 配置
4. 联系项目维护者

---

**注意**：文档系统已经完全配置好，您可以直接使用。所有 API 文档都是基于您的代码自动生成的，包含了详细的功能说明、参数说明、使用示例和最佳实践。
