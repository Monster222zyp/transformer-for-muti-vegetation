# PDF 导出说明

合并版 PDF 与 VS Code 的 Markdown PDF 插件共用项目内的排版设置和样式。Markdown 正文未作修改。

## 在 VS Code 中导出

1. 在 VS Code 中打开整个 `transformer-for-muti-vegetation` 项目文件夹，而不只是单独打开 Markdown 文件。
2. 以文本编辑器打开 `申请书.md`。如果默认显示 Markdown 预览，先切换到源文件编辑界面。
3. 按 `Ctrl+Shift+P`，执行 `Markdown PDF: Export (pdf)`。
4. PDF 输出到 Markdown 所在文件夹，与 Markdown 使用相同的文件名。再次导出会更新同名 PDF；没有启用保存时自动导出。

## 当前排版

- A4 纵向，上边距 22 mm、下边距 20 mm、左右边距 22 mm。
- 中文正文使用宋体，英文使用 Times New Roman；正文 12 pt，行距 1.65 倍，首行缩进两个字符。
- 标题使用微软雅黑，采用黑色、分级字号，不加装饰横线。
- 表格使用浅灰表头和细边框，行高随内容自动展开。
- 页眉不显示日期或文件名，页脚居中显示当前页码与总页数。
- 保留文献链接，但不在纸面额外展开网址。

## 设置位置与范围

页面参数保存在项目根目录的 `.vscode/settings.json`，字体、标题和表格样式保存在 `.vscode/markdown-pdf.css`。CSS 是控制文档外观的样式文件；可修改文件开头的字号和字体变量来调整排版。

这些是项目级设置，未修改 VS Code 的全局用户设置。本项目的本地样式会覆盖全局 `markdown-pdf.styles` 中原有的网络样式和无效示例路径。

插件使用已安装的 `C:/Program Files/Google/Chrome/Application/chrome.exe` 进行渲染。更换电脑或 Chrome 安装位置后，需要更新 `.vscode/settings.json` 中的 `markdown-pdf.executablePath`。字体或浏览器版本变化可能导致少量分页差异。
