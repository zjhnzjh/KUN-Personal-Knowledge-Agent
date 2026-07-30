# Contributing

感谢你关注 KUN Personal Knowledge Agent。

## 开发准备

请先阅读 [README.md](README.md) 与 [AGENTS.md](AGENTS.md)，并确保本机具备：

- Windows 10/11
- Node.js 22.13 或更高版本
- Python 3.11 或更高版本

安装依赖并运行验证：

```powershell
npm install
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
npm test
```

## 提交约定

- 每个提交只处理一个清晰主题，提交信息使用简洁的祈使句。
- 功能变更应补充相应测试，并同步更新 README 或 `docs/`。
- 不要提交 API Key、凭据、私人文件、提取文本、索引、数据库或真实用户数据。
- 文件写入、Memory 变更和联网能力必须继续经过权限层，不得绕过确认。

## Pull Request

Pull Request 应说明变更目的、验证方式、用户可见影响和已知限制。涉及界面时请附截图；涉及 RAG 指标时请同时说明数据集、模型、机器配置和测试日期。

## 许可证

本仓库当前未授予开源许可证。提交贡献前，请先与维护者确认贡献和授权安排。