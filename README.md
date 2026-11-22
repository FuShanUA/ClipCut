# ClipCut Studio

自动化严肃视频切片与精装 Remix 应用（CLI + FastAPI + React/Vite）。

## 快速启动

### 1. 启动 Web 工作台（或直接双击桌面 `ClipCut.app`）
```bash
cd /path/to/ClipCut
./start-local.command
# 浏览器打开 http://127.0.0.1:8790
```

### 2. CLI 一键自动出片
```bash
cd /path/to/ClipCut

# 直接指定视频文件（默认使用百炼：文本 qwen3.8-max + 多模态 qwen3.5-omni-plus）
.venv/bin/clipcut run "/path/to/video.mp4"

# 分别指定文本模型和多模态模型
.venv/bin/clipcut run "/path/to/video.mp4" --text-model qwen4-max --multimodal-model qwen3.8-omni-flash

# 指定目录（单视频自动选中，多视频交互列出供选择）
.venv/bin/clipcut run "/path/to/folder"

# 仅分析与多模态复核（跳过渲染）
.venv/bin/clipcut run "/path/to/video.mp4" --dry-run

# 查看结构化回执日志
.venv/bin/clipcut logs "/path/to/video.mp4"

# 增量渲染变动项 / 重排 Remix
.venv/bin/clipcut render "/path/to/video.mp4"
.venv/bin/clipcut replan-remix "/path/to/video.mp4"

# 按切片修复字幕并只重烧该切片；如需全片硬字幕同步修正，追加 --render-full-video
.venv/bin/clipcut repair-subtitle "/path/to/video.mp4" \
  --clip-id clip_10 \
  --repair '{"utterance_id":193,"zh":"正确中文","en":"Correct English"}'
```

## 配置文件

- [config/editorial.yaml](config/editorial.yaml)：中国热点词表、主话题去重开关、切片时长范围、模型路由与 Remix 混剪参数（支持在 Web 界面右上角「热点规则 YAML」实时修改）。
- `config/local.yaml`（参考 [config/local.example.yaml](config/local.example.yaml) 复制，已被 `.gitignore` 忽略）：本地视频号 / B站 / 小红书上传脚本、登录态目录与 AutoSub `cover_utils.py` 路径（也可用 `CLIPCUT_*` 环境变量覆盖）。
- 产物位置：
  - 中间状态与回执日志：`<视频所在目录>/.clipcut/<project_id>/`（含 `project.json`、`receipts.jsonl`、`pipeline.log`、`undo/`）
  - 成品输出：`<视频所在目录>/clips/`（切片 `.mp4`、封面 `.jpg`、`00_Remix_高光集锦.mp4`、`clips_index.md`、`clips_detail.md`、`remix_script.md`、`baocut_export_plan.json`）
