# PDF 转换与插件配置记录

- 转换时输入为 `文档/项目申请书/申请书_合并版.md`，输出为同目录的 `申请书_合并版.pdf`。
- 最终核对时文件已分别重命名为 `申请书.md` 与 `申请书.pdf`；保留当前名称，PDF 为 A4，共 7 页。
- 复用本机安装的 Markdown PDF 2.2.0 的 Markdown 解析器、HTML 模板、PDF 参数构造和 Puppeteer 驱动；通过内存载入提供转换所需的只读 VS Code 接口，没有修改插件安装文件，也没有依赖项目 Python 环境。
- 修改范围：项目 `.vscode/settings.json`，新增 `.vscode/markdown-pdf.css`；保留原有 Python 设置，不改动全局用户设置。
- 实际样式校验：正文 16 px，即 12 pt；字体 Times New Roman / SimSun；总标题 29.3333 px，即 22 pt。
- PDF 中 92 个正文、标题、列表和表格内容块均通过文本完整性核对，已接收状态保留，链接注释保留。
- 全部 7 页已渲染成图片逐页检查；未发现乱码、文字截断、重叠或表格溢出。
- 输入 Markdown 的 SHA-256 转换前后相同：`3749D2917CCF844F471D67A6F829AEFDBCE08C8AF7C19F860DA70615213B3E90`。
- 验证使用插件内部转换流程，而不是通过 VS Code 界面点击导出；设置加载路径和实际生效样式已核对。
- 临时渲染脚本已删除；HTML 和页面检查图片暂留系统临时目录 `C:/Users/Mayn/AppData/Local/Temp/codex-proposal-pdf-jOfxDn`，自动清理受到运行环境限制。项目内保留正式 PDF、样式、导出说明及本记录。
