from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

from .adapters.ingest import resolve_work_dir, scan_input_path
from .pipeline.orchestrator import (
    ALL_STAGES,
    find_existing_plan,
    run_existing_project_stages,
    run_pipeline,
)
from .publish import publish_to_platforms
from .receipts import ProjectStore
from .models import SubtitleRepairItem
from .subtitle_repair import apply_subtitle_repairs

app = typer.Typer(name="clipcut", help="ClipCut — 严肃自动化视频切片与精装 Remix 工具")
console = Console()


def _select_video_from_input(path: str, non_interactive: bool = False) -> Path:
    res = scan_input_path(path)
    if not res.videos:
        console.print(f"[red]目录内未找到支持的视频文件: {path}[/red]")
        raise typer.Exit(code=1)
    if len(res.videos) == 1:
        return Path(res.videos[0].path)

    table = Table(title=f"发现 {len(res.videos)} 个视频文件，请选择目标视频")
    table.add_column("编号", justify="right", style="cyan")
    table.add_column("文件名", style="bold")
    table.add_column("分辨率")
    table.add_column("时长")
    table.add_column("大小")
    table.add_column("标签")
    for idx, v in enumerate(res.videos, start=1):
        size_mb = v.size_bytes / (1024 * 1024)
        mins, secs = divmod(int(v.duration_s), 60)
        tag = "硬字幕" if v.is_hardsub else "原片"
        table.add_row(str(idx), v.filename, f"{v.width}x{v.height}", f"{mins:02d}:{secs:02d}", f"{size_mb:.1f} MB", tag)
    console.print(table)

    if non_interactive:
        console.print(f"非交互模式，默认选择第 1 个: {res.videos[0].filename}")
        return Path(res.videos[0].path)

    choice = IntPrompt.ask("输入视频编号", default=1)
    if choice < 1 or choice > len(res.videos):
        console.print("[red]编号超出范围[/red]")
        raise typer.Exit(code=1)
    return Path(res.videos[choice - 1].path)


@app.command()
def scan(path: str = typer.Argument(..., help="视频文件或目录路径")):
    """扫描文件或目录内的视频、字幕和元数据文件"""
    res = scan_input_path(path)
    console.print(f"路径: {res.input_path} ({'目录' if res.is_directory else '单文件'})", markup=False)
    for idx, v in enumerate(res.videos, start=1):
        console.print(
            f"  [{idx}] {v.filename} | {v.width}x{v.height} | {v.duration_s:.1f}s | {v.size_bytes / 1048576:.1f}MB"
            f" | 字幕: {Path(v.suggested_subtitle).name if v.suggested_subtitle else '无'}"
            f" | 元数据: {Path(v.suggested_info_json).name if v.suggested_info_json else '无'}",
            markup=False,
        )


@app.command()
def plan(
    path: str = typer.Argument(..., help="视频文件或包含视频的目录路径"),
    subtitle: Optional[str] = typer.Option(None, "--subtitle", "-s", help="显式指定字幕文件 (.srt/.vtt)"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="原视频链接（用于拉取标题/简介/章节）"),
    title: Optional[str] = typer.Option(None, "--title", help="手动覆盖原视频标题"),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="文本分析模型"),
    multimodal_model: Optional[str] = typer.Option(None, "--multimodal-model", help="多模态复核模型"),
    force: bool = typer.Option(False, "--force", help="忽略已完成策划并强制重跑"),
    yes: bool = typer.Option(False, "--yes", "-y", help="目录含多视频时不交互提示，默认选第一个"),
):
    """【推荐】制定切片策略与Remix规划（不执行重型渲染，产出后供人工审阅选择）"""
    video_path = _select_video_from_input(path, non_interactive=yes)
    stages = ["ingest", "signals", "scan_cuts", "analyze", "verify_and_plan"]

    console.print(f"目标视频: {video_path.name}", markup=False)
    if not force:
        existing = find_existing_plan(
            video_path,
            subtitle_path=subtitle,
            expected_title=title,
        )
        if existing is not None:
            console.print("[bold green]检测到已完成策划，直接复用。[/bold green]")
            console.print(f"项目状态: {existing.source.work_dir}/project.json")
            return
    console.print(f"执行阶段: {' -> '.join(stages)} (仅规划，跳过渲染)")

    manifest, store = run_pipeline(
        video_path=video_path,
        subtitle_path=subtitle,
        source_url=url,
        manual_title=title,
        stages=stages,
        text_model=text_model,
        multimodal_model=multimodal_model,
    )

    table = Table(title=f"切片策略规划完成 (共 {len(manifest.clips)} 支独立切片 + {len(manifest.remix.items)} 段 Remix)")
    table.add_column("#", justify="right")
    table.add_column("预选")
    table.add_column("评分")
    table.add_column("标题")
    table.add_column("类别")
    table.add_column("区间")
    table.add_column("时长")
    for c in manifest.clips:
        enabled_tag = "[green]✓ 预选[/green]" if c.enabled else "[dim]未启用[/dim]"
        table.add_row(
            f"{c.index:02d}",
            enabled_tag,
            f"{getattr(c, 'score', 7.5):.1f}",
            c.title,
            ",".join(c.categories),
            f"{int(c.start_s // 60):02d}:{int(c.start_s % 60):02d}-{int(c.end_s // 60):02d}:{int(c.end_s % 60):02d}",
            f"{c.end_s - c.start_s:.0f}s",
        )
    console.print(table)
    console.print(f"\n[bold green]切片策略制定完成！当前未渲染任何视频。[/bold green]")
    console.print(f"👉 推荐运行 [cyan]clipcut serve[/cyan] 打开网页工作台人工勾选、微调并按需渲染；")
    console.print(f"👉 或直接运行 [cyan]clipcut render \"{video_path}\"[/cyan] 渲染当前预选的切片。")


@app.command()
def run(
    path: str = typer.Argument(..., help="视频文件或包含视频的目录路径"),
    subtitle: Optional[str] = typer.Option(None, "--subtitle", "-s", help="显式指定字幕文件 (.srt/.vtt)"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="原视频链接（用于拉取标题/简介/章节）"),
    title: Optional[str] = typer.Option(None, "--title", help="手动覆盖原视频标题"),
    stage: Optional[str] = typer.Option(None, "--stage", help="仅运行指定阶段，逗号分隔，如 signals,analyze"),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="文本分析模型（如 qwen4-max / dashscope:qwen3.8-max / gemini:gemini-2.5-flash）"),
    multimodal_model: Optional[str] = typer.Option(None, "--multimodal-model", help="多模态复核模型（如 qwen3.5-omni-plus / qwen3.8-omni-flash / gemini:gemini-2.5-flash）"),
    dry_run: bool = typer.Option(False, "--dry-run", "--plan-only", help="仅分析与规划，跳过最终 ffmpeg 渲染"),
    force_render: bool = typer.Option(False, "--force-render", help="忽略增量哈希缓存强制重渲"),
    yes: bool = typer.Option(False, "--yes", "-y", help="目录含多视频时不交互提示，默认选第一个"),
):
    """一键运行完整五阶段切片与精装 Remix 流水线"""
    video_path = _select_video_from_input(path, non_interactive=yes)
    if stage:
        stages = [s.strip() for s in stage.split(",") if s.strip() in ALL_STAGES]
    elif dry_run:
        stages = ["ingest", "signals", "analyze", "verify_and_plan"]
    else:
        stages = ALL_STAGES

    console.print(f"目标视频: {video_path.name}", markup=False)
    console.print(f"执行阶段: {' -> '.join(stages)}")

    manifest, store = run_pipeline(
        video_path=video_path,
        subtitle_path=subtitle,
        source_url=url,
        manual_title=title,
        stages=stages,
        force_render=force_render,
        text_model=text_model,
        multimodal_model=multimodal_model,
    )

    table = Table(title=f"切片结果 ({len(manifest.clips)} 支独立切片 + {len(manifest.remix.items)} 段 Remix)")
    table.add_column("#", justify="right")
    table.add_column("标题")
    table.add_column("类别")
    table.add_column("区间")
    table.add_column("时长")
    table.add_column("复核")
    for c in manifest.clips:
        table.add_row(
            f"{c.index:02d}",
            c.title,
            ",".join(c.categories),
            f"{int(c.start_s // 60):02d}:{int(c.start_s % 60):02d}-{int(c.end_s // 60):02d}:{int(c.end_s % 60):02d}",
            f"{c.end_s - c.start_s:.0f}s",
            c.multimodal_status,
        )
    console.print(table)
    console.print(f"项目状态: {store.manifest_path}")
    console.print(f"审计日志: {store.receipts_path}")
    console.print(f"成品目录: {manifest.source.output_dir}")


@app.command()
def render(
    path: str = typer.Argument(..., help="视频文件路径"),
    force: bool = typer.Option(False, "--force", help="强制全部重渲"),
):
    """按增量哈希仅重渲有改动的切片和 Remix"""
    vpath = _select_video_from_input(path, non_interactive=True)
    work_dir = resolve_work_dir(vpath)
    manifest = run_existing_project_stages(work_dir, ["render"], force_render=force)
    console.print(f"渲染完成，输出目录: {manifest.source.output_dir}")


@app.command("replan-remix")
def replan_remix(
    path: str = typer.Argument(..., help="视频文件路径"),
    text_model: Optional[str] = typer.Option(None, "--text-model", help="可选覆盖 Remix 策展文本模型"),
):
    """基于当前启用切片重新策划 Remix 混剪"""
    vpath = _select_video_from_input(path, non_interactive=True)
    work_dir = resolve_work_dir(vpath)
    manifest = run_existing_project_stages(work_dir, ["replan_remix"], text_model=text_model)
    console.print(f"Remix 重新编排完成: {len(manifest.remix.items)} 段")


@app.command("repair-subtitle")
def repair_subtitle(
    path: str = typer.Argument(..., help="视频文件路径"),
    clip_id: str = typer.Option(..., "--clip-id", help="目标切片 ID，如 clip_10"),
    repair: list[str] = typer.Option(..., "--repair", help='JSON 修复项，如 \'{"utterance_id":193,"zh":"...","en":"..."}\'，可重复'),
    rebuild_full_subtitle: bool = typer.Option(True, "--rebuild-full-subtitle/--no-rebuild-full-subtitle", help="是否同步重建中英/双语字幕文件"),
    render_clip: bool = typer.Option(True, "--render-clip/--no-render-clip", help="修复后是否只重烧该切片"),
    render_full_video: bool = typer.Option(False, "--render-full-video", help="修复后是否重烧全片硬字幕"),
):
    """按切片修复字幕文本，并可选重烧该切片或全片硬字幕"""
    import json

    vpath = _select_video_from_input(path, non_interactive=True)
    work_dir = resolve_work_dir(vpath)
    store = ProjectStore(work_dir)
    manifest = store.load()

    items: list[SubtitleRepairItem] = []
    for raw in repair:
        try:
            payload = json.loads(raw)
            items.append(SubtitleRepairItem.model_validate(payload))
        except Exception as exc:
            console.print(f"[red]修复项 JSON 解析失败: {raw}[/red]")
            raise typer.Exit(code=1) from exc

    try:
        outcome = apply_subtitle_repairs(
            manifest,
            store,
            clip_id,
            items,
            rebuild_full_subtitle=rebuild_full_subtitle,
            render_clip=render_clip,
            render_full_video=render_full_video,
        )
    except Exception as exc:
        console.print(f"[red]字幕修复失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]已修复 {len(outcome.changed_utterance_ids)} 句字幕。[/green]")
    console.print(f"双语字幕: {outcome.bi_srt_path}")
    if rebuild_full_subtitle:
        console.print(f"中文字幕: {outcome.cn_srt_path}")
        console.print(f"英文字幕: {outcome.en_srt_path}")
    console.print(f"ASS 字幕: {outcome.ass_path}")
    if outcome.clip_output_path:
        console.print(f"切片输出: {outcome.clip_output_path}")
    if outcome.full_hardsub_path:
        console.print(f"全片硬字幕: {outcome.full_hardsub_path}")


@app.command()
def publish(
    path: str = typer.Argument(..., help="视频文件路径"),
    target: str = typer.Option("remix", "--target", help="发布目标：remix 或 clip_01 等"),
    platforms: str = typer.Option("wechat", "--platforms", help="逗号分隔平台：bilibili,wechat,xiaohongshu"),
    title: Optional[str] = typer.Option(None, "--title", help="覆盖发布标题"),
    cover_title: Optional[str] = typer.Option(None, "--cover-title", help="覆盖封面标题"),
    desc: Optional[str] = typer.Option(None, "--desc", help="覆盖简介文案"),
    tags: Optional[str] = typer.Option(None, "--tags", help="逗号分隔标签"),
    public: bool = typer.Option(False, "--public", help="公开发布（默认安全存草稿/仅自己可见）"),
):
    """复用 AutoSub 发布链将指定切片或 Remix 发布到视频号/B站/小红书"""
    vpath = _select_video_from_input(path, non_interactive=True)
    store = ProjectStore(resolve_work_dir(vpath))
    plat_list = [p.strip().lower() for p in platforms.split(",") if p.strip()]
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    manifest = publish_to_platforms(
        store=store,
        target=target,
        platforms=plat_list,
        title=title,
        cover_title=cover_title,
        desc=desc,
        tags=tag_list,
        is_public=public,
    )
    recs = manifest.remix.publish_records if target == "remix" else next(
        (c.publish_records for c in manifest.clips if c.clip_id == target), []
    )
    for r in recs:
        console.print(f"[{r.platform}] {r.status} — {r.message} {r.url_or_id or ''} {r.error or ''}")


@app.command()
def logs(
    path: str = typer.Argument(..., help="视频文件路径"),
    stage: Optional[str] = typer.Option(None, "--stage", help="按阶段筛选 receipts"),
    limit: int = typer.Option(40, "--limit", "-n", help="显示条数"),
):
    """查看项目 Receipts 决策日志"""
    vpath = _select_video_from_input(path, non_interactive=True)
    store = ProjectStore(resolve_work_dir(vpath))
    receipts = store.list_receipts(stage=stage, limit=limit)
    for r in receipts:
        console.print(f"[{r.stage}/{r.actor}] {r.action} ({r.elapsed_ms}ms) — {r.summary}")
        if r.error:
            console.print(f"  [red]ERROR: {r.error}[/red]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8790, "--port", "-p"),
):
    """启动本地 Web 检视与微调工作台"""
    import uvicorn

    console.print(f"启动 ClipCut 工作台: http://{host}:{port}")
    uvicorn.run("clipcut.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
