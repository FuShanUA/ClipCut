from __future__ import annotations

import codecs
import concurrent.futures
import json
import os
import re
import select
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import load_editorial_config, load_local_config
from ..models import ProjectManifest, PublishRecord, utc_now_iso
from ..receipts import ProjectStore
from ..render.renderer import sync_platform_covers_for_target

PLATFORM_LABELS = {
    "wechat": "微信视频号",
    "bilibili": "B站",
    "xiaohongshu": "小红书",
}

_UNCONFIGURED_HINT = "未配置本地上传脚本（请在 config/local.yaml 或环境变量中配置）"
_ORIG_SUBPROCESS_RUN = subprocess.run


def _opt_path(raw: Any) -> Path | None:
    s = str(raw or "").strip()
    return Path(s).expanduser() if s else None


def _resolve_python(*candidates: str) -> str | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        w = shutil.which(c)
        if w:
            return w
    return None


def _get_paths() -> dict[str, dict[str, Any]]:
    local_pub = (load_local_config().get("publishers") or {})
    wx_cfg = local_pub.get("wechat") or {}
    bi_cfg = local_pub.get("bilibili") or {}
    xhs_cfg = local_pub.get("xiaohongshu") or {}

    wx_script = _opt_path(os.environ.get("CLIPCUT_WECHAT_SCRIPT") or wx_cfg.get("script"))
    wx_profile = _opt_path(os.environ.get("WECHAT_BROWSER_DATA_DIR") or wx_cfg.get("profile_dir")) or (
        wx_script.parent.parent / "assets" / "browser_data" if wx_script else None
    )

    bili_script = _opt_path(os.environ.get("CLIPCUT_BILI_SCRIPT") or bi_cfg.get("script"))
    bili_login = _opt_path(os.environ.get("CLIPCUT_BILI_LOGIN_SCRIPT") or bi_cfg.get("login_script")) or (
        bili_script.parent / "login.py" if bili_script else None
    )
    bili_cookies = _opt_path(os.environ.get("CLIPCUT_BILI_COOKIES") or bi_cfg.get("cookies_file")) or (
        bili_script.parent.parent / "assets" / "cookies.json" if bili_script else None
    )

    xhs_script = _opt_path(os.environ.get("CLIPCUT_XHS_SCRIPT") or xhs_cfg.get("script"))
    xhs_login = _opt_path(os.environ.get("CLIPCUT_XHS_LOGIN_SCRIPT") or xhs_cfg.get("login_script")) or (
        xhs_script.parent / "login_xhs.py" if xhs_script else None
    )
    xhs_cookies = _opt_path(os.environ.get("CLIPCUT_XHS_COOKIES") or xhs_cfg.get("cookies_file")) or (
        xhs_script.parent.parent / "assets" / "cookies_xhs.json" if xhs_script else None
    )

    return {
        "wechat": {
            "label": "微信视频号",
            "python": _resolve_python(
                os.environ.get("CLIPCUT_WECHAT_PYTHON", ""),
                str(wx_cfg.get("python") or ""),
                "python3",
            ),
            "script": wx_script,
            "login_cmd_args": [str(wx_script), "--login"] if wx_script else [],
            "profile_dir": wx_profile,
        },
        "bilibili": {
            "label": "B站",
            "python": _resolve_python(
                os.environ.get("CLIPCUT_BILI_PYTHON", ""),
                str(bi_cfg.get("python") or ""),
                "python3",
            ),
            "script": bili_script,
            "login_script": bili_login,
            "cookies_file": bili_cookies,
        },
        "xiaohongshu": {
            "label": "小红书",
            "python": _resolve_python(
                os.environ.get("CLIPCUT_XHS_PYTHON", ""),
                str(xhs_cfg.get("python") or ""),
                "python3",
            ),
            "script": xhs_script,
            "login_script": xhs_login,
            "cookies_file": xhs_cookies,
        },
    }


def _check_bili_cookies(cookies_path: Path | None) -> tuple[bool, str]:
    if cookies_path is None:
        return False, _UNCONFIGURED_HINT
    if not cookies_path.is_file():
        return False, "未找到 B站 cookies.json，请先扫码登录"
    try:
        data = json.loads(cookies_path.read_text(encoding="utf-8"))
        req = ("SESSDATA", "bili_jct", "buvid3")
        missing = [k for k in req if not str(data.get(k) or "").strip()]
        if missing:
            return False, f"B站 Cookie 缺少字段: {', '.join(missing)}"
        return True, "B站登录态已就绪"
    except Exception as exc:
        return False, f"读取 B站 Cookie 失败: {exc}"


def _check_xhs_cookies(cookies_path: Path | None) -> tuple[bool, str]:
    if cookies_path is None:
        return False, _UNCONFIGURED_HINT
    if not cookies_path.is_file() or cookies_path.stat().st_size < 16:
        return False, "未找到小红书 cookies_xhs.json，请先扫码登录"
    try:
        data = json.loads(cookies_path.read_text(encoding="utf-8"))
        if not data:
            return False, "小红书 Cookie 为空，请重新登录"
        return True, "小红书登录态已就绪"
    except Exception as exc:
        return False, f"读取小红书 Cookie 失败: {exc}"


def get_platforms_status() -> dict[str, dict[str, Any]]:
    paths = _get_paths()
    out: dict[str, dict[str, Any]] = {}

    # 1. WeChat Channels
    wx = paths["wechat"]
    wx_script = wx.get("script")
    wx_prof = wx.get("profile_dir")
    wx_script_ready = bool(wx_script and wx_script.is_file())
    wx_logged = bool(wx_prof and wx_prof.is_dir() and any(wx_prof.iterdir()))
    if not wx_script_ready and not wx_logged:
        wx_hint = _UNCONFIGURED_HINT
    else:
        wx_hint = "微信视频号浏览器登录态已就绪" if wx_logged else "尚未扫码登录微信视频号"
    out["wechat"] = {
        "platform": "wechat",
        "label": "微信视频号",
        "logged_in": wx_logged,
        "python_ready": bool(wx["python"]) and wx_script_ready,
        "script": str(wx_script) if wx_script else "",
        "hint": wx_hint,
    }

    # 2. Bilibili
    bi = paths["bilibili"]
    bi_script = bi.get("script")
    bi_script_ready = bool(bi_script and bi_script.is_file())
    bi_logged, bi_hint = _check_bili_cookies(bi.get("cookies_file"))
    if not bi_script_ready:
        bi_hint = _UNCONFIGURED_HINT
    out["bilibili"] = {
        "platform": "bilibili",
        "label": "B站",
        "logged_in": bi_logged,
        "python_ready": bool(bi["python"]) and bi_script_ready,
        "script": str(bi_script) if bi_script else "",
        "hint": bi_hint,
    }

    # 3. Xiaohongshu
    xhs = paths["xiaohongshu"]
    xhs_script = xhs.get("script")
    xhs_script_ready = bool(xhs_script and xhs_script.is_file())
    xhs_logged, xhs_hint = _check_xhs_cookies(xhs.get("cookies_file"))
    if not xhs_script_ready:
        xhs_hint = _UNCONFIGURED_HINT
    out["xiaohongshu"] = {
        "platform": "xiaohongshu",
        "label": "小红书",
        "logged_in": xhs_logged,
        "python_ready": bool(xhs["python"]) and xhs_script_ready,
        "script": str(xhs_script) if xhs_script else "",
        "hint": xhs_hint,
    }
    return out


def launch_platform_login(platform: str) -> dict[str, Any]:
    """Spawn the interactive QR login helper in a background process (non-blocking)."""
    paths = _get_paths()
    if platform not in paths:
        raise ValueError(f"不支持的平台: {platform}")
    info = paths[platform]
    py = info.get("python")
    if not py:
        raise RuntimeError(f"未找到 {PLATFORM_LABELS.get(platform, platform)} 所需的 Python 运行环境")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if platform == "wechat":
        script = info.get("script")
        if not script or not script.is_file():
            raise FileNotFoundError(f"未找到视频号脚本，请在 config/local.yaml 配置 ({script or '未设置'})")
        if info.get("profile_dir") is not None:
            env["WECHAT_BROWSER_DATA_DIR"] = str(info["profile_dir"])
        cmd = [py, "-u", str(script), "--login"]
    elif platform == "bilibili":
        script = info.get("login_script")
        if not script or not script.is_file():
            raise FileNotFoundError(f"未找到 B站登录脚本，请在 config/local.yaml 配置 ({script or '未设置'})")
        cmd = [py, "-u", str(script)]
    else:
        script = info.get("login_script")
        if not script or not script.is_file():
            raise FileNotFoundError(f"未找到小红书登录脚本，请在 config/local.yaml 配置 ({script or '未设置'})")
        cmd = [py, "-u", str(script)]

    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    return {
        "started": True,
        "platform": platform,
        "pid": proc.pid,
        "message": f"已唤起 {PLATFORM_LABELS.get(platform, platform)} 扫码登录窗口，请在弹出窗口完成扫码",
    }


def _upsert_record(records: list[PublishRecord], rec: PublishRecord) -> None:
    for idx, existing in enumerate(records):
        if existing.platform == rec.platform:
            records[idx] = rec
            return
    records.append(rec)


def _update_target_record(store: ProjectStore, target: str, record: PublishRecord) -> None:
    """Thread-safely update a platform's PublishRecord inside project.json."""
    def _mutate(m: ProjectManifest) -> None:
        _, _, _, _, _, rec_list = _default_target_metadata(m, target)
        _upsert_record(rec_list, record)

    store.update_manifest(_mutate)


def _parse_publish_progress(
    plat: str,
    line: str,
    prev_pct: float | None,
    prev_stage: str,
) -> tuple[float | None, str, str] | None:
    """Parse platform uploader stdout/stderr line into (pct, short_stage, detail_message)."""
    clean = line.strip()
    if not clean:
        return None
    floor = prev_pct or 0.0

    if plat == "bilibili":
        m = re.search(r"(?:上传进度|Progress):\s*([\d.]+)%", clean)
        if m:
            raw = max(0.0, min(100.0, float(m.group(1))))
            if raw >= 99.0 and prev_stage in ("提交分P", "处理封面", "传封面", "封面完成", "提交发布"):
                return None
            return min(95.0, round(raw, 1)), "上传中", f"B站视频上传中 {raw:.1f}%"
        if "分块传输完成" in clean or "提交分P元数据" in clean:
            return max(floor, 95.0), "提交分P", "分块传输完成，提交分P元数据..."
        if "分块上传完成" in clean or "进入封面" in clean:
            return max(floor, 96.0), "处理封面", "分块上传完成，准备上传封面..."
        if "正在上传B站封面" in clean:
            return max(floor, 97.0), "传封面", "正在上传B站封面..."
        if "B站封面上传完成" in clean:
            return max(floor, 98.0), "封面完成", "B站封面上传完成"
        if "正在提交B站发布请求" in clean or "B站发布请求已返回" in clean:
            return max(floor, 99.0), "提交发布", clean.lstrip("> ")
        if any(k in clean for k in ("线路探测", "封面已选择", "获取合集", "获取到")):
            return max(floor, 5.0), "准备中", clean.lstrip("> ")
        return None

    if plat == "wechat":
        if "另一个微信上传/登录进程正在使用浏览器 profile" in clean:
            return max(floor, 2.0), "排队锁", "另一个微信上传进程占用浏览器 profile，排队等待中..."
        m_wait = re.search(r"仍在等待微信浏览器 profile 锁[（(](\d+)\s*秒", clean)
        if m_wait:
            secs = m_wait.group(1)
            return max(floor, 2.0), f"排队锁{secs}s", f"排队等待微信浏览器 profile 锁（已等 {secs} 秒）"
        if "已获得微信浏览器 profile 锁" in clean:
            return max(floor, 5.0), "启浏览器", "已获得浏览器锁，启动微信创作者页面..."
        m_open = re.search(r"打开发布页面\s*\(尝试\s*(\d+)/(\d+)\)", clean)
        if m_open:
            attempt = int(m_open.group(1))
            stage = "打开页面" if attempt == 1 else f"重试{m_open.group(1)}/{m_open.group(2)}"
            return (10.0 if attempt == 1 else max(floor, 10.0)), stage, clean.lstrip("> ")
        if "等待页面加载" in clean:
            return max(floor, 12.0), "加载页面", "等待视频号发布页加载..."
        if "选择视频文件" in clean or "视频文件已选择" in clean:
            return max(floor, 15.0), "开始上传", clean.lstrip("> ")
        m_pct = re.search(r">>>\s*([^:\r\n]+?):\s*(\d+(?:\.\d+)?)%", clean)
        if m_pct:
            raw_stage = m_pct.group(1).strip()
            raw = max(0.0, min(100.0, float(m_pct.group(2))))
            if "转码" in raw_stage or "处理" in raw_stage:
                pct = round(75.0 + raw * 0.12, 1)
                return max(floor, pct), "转码中", f"微信视频{raw_stage} {raw:.0f}%"
            pct = round(15.0 + raw * 0.60, 1)
            return max(floor, pct), "上传中", f"微信视频{raw_stage} {raw:.0f}%"
        if "上传完成！" in clean or "上传应已完成" in clean:
            return max(floor, 76.0), "填信息", "视频上传完成，正在填写标题与描述..."
        if any(k in clean for k in ("已填写标题", "已填写描述")) or re.search(r"已添加\s*\d+\s*个标签", clean):
            return max(floor, 80.0), "填信息", clean.lstrip("> ")
        if "微信位置、视频标注、原创标志均已设置" in clean:
            return max(floor, 83.0), "发布设置", clean.lstrip("> ")
        if any(
            k in clean
            for k in (
                "等待封面与分享卡片生成",
                "待生成资源",
                "封面编辑器仍在处理",
                "等待封面编辑器",
            )
        ):
            return max(floor, 85.0), "等封面", clean.lstrip("> ")
        if "已有草稿缺失封面" in clean or "开始自动补修" in clean:
            return max(floor, 88.0), "补封面", clean.lstrip("> ")
        if "封面与分享卡片生成完成" in clean or "封面编辑器已可编辑" in clean:
            return max(floor, 88.0), "设封面", clean.lstrip("> ")
        if "封面未生效，关闭弹层等待" in clean:
            return max(floor, 89.0), "重试封面", clean.lstrip("> ")
        if "横竖双封面均已回读确认" in clean:
            return max(floor, 94.0), "封面完成", clean.lstrip("> ")
        if any(k in clean for k in ("点击存草稿按钮", "点击发表按钮", "点击发布按钮")):
            return max(floor, 96.0), "提交中", clean.lstrip("> ")
        if "等待发布完成" in clean or "正在到草稿箱核验" in clean:
            return max(floor, 98.0), "核验中", clean.lstrip("> ")
        if re.search(r"尝试\s*\d+\s*失败", clean):
            return prev_pct if prev_pct is not None else 10.0, "重试中", clean.lstrip("> ")
        return None

    if plat == "xiaohongshu":
        if "正在启动" in clean or "成功接管真实 Chrome" in clean:
            return max(floor, 8.0), "启浏览器", clean
        if "正在打开小红书视频发布页面" in clean:
            return max(floor, 15.0), "打开页面", clean
        if "正在上传视频文件" in clean or "视频已选择" in clean:
            return max(floor, 25.0), "上传视频", clean
        if "填写标题" in clean:
            return max(floor, 35.0), "填标题", clean
        if "填写描述" in clean or "描述填写成功" in clean:
            return max(floor, 42.0), "填描述", clean
        if "正在等待视频上传与分析完成" in clean:
            return max(floor, 50.0), "上传分析", clean
        m_wait = re.search(r"已等待\s*(\d+)\s*秒", clean)
        if m_wait:
            secs = int(m_wait.group(1))
            pct = min(82.0, round(50.0 + secs * 0.4, 1))
            return max(floor, pct), f"分析{secs}s", f"小红书视频上传与分析中（已等待 {secs} 秒）"
        if "视频上传成功" in clean:
            return max(floor, 85.0), "上传完成", clean
        if "正在上传配套封面" in clean:
            return max(floor, 88.0), "传封面", clean
        if "封面上传成功" in clean:
            return max(floor, 92.0), "封面完成", clean
        if "正在设置加入合集" in clean or "正在设置【原创声明】" in clean:
            return max(floor, 94.0), "发布设置", clean
        if "正在提交【公开正式发布】" in clean or "正在保存至【草稿箱】" in clean:
            return max(floor, 97.0), "提交中", clean
        if "Keeping browser open" in clean:
            return max(floor, 99.0), "收尾中", "小红书已提交，等待浏览器收尾..."
        return None

    return None


def _run_publish_process(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: float,
    on_line: Callable[[str], None],
) -> tuple[int, list[str]]:
    """Run uploader subprocess while streaming both CR (\r) and LF (\n) progress lines."""
    if subprocess.run is not _ORIG_SUBPROCESS_RUN:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
        combined = f"{res.stdout or ''}\n{res.stderr or ''}"
        out_lines: list[str] = []
        for seg in re.split(r"[\r\n]", combined):
            s = seg.strip()
            if s:
                out_lines.append(s)
                on_line(s)
        return res.returncode, out_lines

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    lines: list[str] = []
    start = time.monotonic()

    def _emit_segment(seg: str) -> None:
        s = seg.strip()
        if s:
            lines.append(s)
            on_line(s)

    try:
        while True:
            elapsed = time.monotonic() - start
            remaining = timeout_s - elapsed
            if remaining <= 0:
                proc.kill()
                proc.wait(timeout=5)
                raise subprocess.TimeoutExpired(cmd, timeout_s)

            ready, _, _ = select.select([fd], [], [], min(1.0, remaining))
            if ready:
                chunk = os.read(fd, 4096)
                if not chunk:
                    buf += decoder.decode(b"", final=True)
                    if buf:
                        for seg in re.split(r"[\r\n]", buf):
                            _emit_segment(seg)
                        buf = ""
                    break
                buf += decoder.decode(chunk)
                if "\r" in buf or "\n" in buf:
                    parts = re.split(r"[\r\n]", buf)
                    for seg in parts[:-1]:
                        _emit_segment(seg)
                    buf = parts[-1]
            elif proc.poll() is not None:
                # Drain any buffered bytes after child exits
                while True:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    buf += decoder.decode(chunk)
                buf += decoder.decode(b"", final=True)
                if buf:
                    for seg in re.split(r"[\r\n]", buf):
                        _emit_segment(seg)
                    buf = ""
                break
        rc = proc.wait(timeout=10)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
    return rc, lines


def _default_target_metadata(manifest: ProjectManifest, target: str) -> tuple[Path | None, str, str, str, list[str], list[PublishRecord]]:
    if target == "remix":
        rmx = manifest.remix
        vpath = Path(rmx.output_file) if rmx.output_file else None
        title = (rmx.title or manifest.metadata.original_theme_zh or rmx.cover_title or "高光混剪").strip()
        cover_title = (rmx.cover_title or title).strip()
        summary = " | ".join(it.card_title for it in rmx.items[:4] if it.enabled and it.card_title)
        desc = f"{manifest.metadata.guest_zh or ''} · {manifest.metadata.channel_zh or manifest.metadata.channel or ''}\n{summary}".strip()
        tags = list(dict.fromkeys(manifest.primary_topics + ["AI", "深度访谈"]))[:6]
        return vpath, title, cover_title, desc, tags, rmx.publish_records

    if target == "compilation":
        cmp = manifest.compilation
        vpath = Path(cmp.output_file) if cmp.output_file else None
        title = (cmp.title or manifest.metadata.original_theme_zh or cmp.cover_title or "切片精选集").strip()
        cover_title = (cmp.cover_title or title).strip()
        enabled_clips = [c for c in manifest.clips if c.enabled]
        summary = " | ".join(c.title for c in enabled_clips[:4])
        desc = f"{manifest.metadata.guest_zh or ''} · {manifest.metadata.channel_zh or manifest.metadata.channel or ''}\n精选 {len(enabled_clips)} 支独立切片：\n{summary}".strip()
        tags = list(dict.fromkeys(manifest.primary_topics + ["AI", "深度访谈", "合集"]))[:6]
        return vpath, title, cover_title, desc, tags, cmp.publish_records

    for c in manifest.clips:
        if c.clip_id == target:
            vpath = Path(c.output_file) if c.output_file else None
            title = (c.title or c.cover_title).strip()
            cover_title = (c.cover_title or c.title).strip()
            desc = (c.summary or c.reason or title).strip()
            tags = list(dict.fromkeys(manifest.primary_topics + c.matched_hot_topics + ["AI", "深度访谈"]))[:6]
            return vpath, title, cover_title, desc, tags, c.publish_records

    raise ValueError(f"未找到目标切片或 Remix: {target}")


def publish_to_platforms(
    *,
    store: ProjectStore,
    target: str,
    platforms: list[str],
    title: str | None = None,
    cover_title: str | None = None,
    desc: str | None = None,
    tags: list[str] | None = None,
    is_public: bool = False,
    collection: str | None = None,
    tid: int | None = None,
    run_id: str | None = None,
    ui_schema: int | None = None,
) -> ProjectManifest:
    """Publish rendered ClipCut MP4 + companion cover JPG to selected platforms via AutoSub scripts."""
    manifest = store.load()
    vpath, def_title, def_cover_title, def_desc, def_tags, rec_list = _default_target_metadata(manifest, target)
    if vpath is None or not vpath.exists():
        raise FileNotFoundError("目标视频尚未渲染，请先执行渲染后再发布")

    req_title = (title or "").strip()
    req_cover = (cover_title or "").strip()
    if (
        ui_schema is None
        and req_title
        and req_title == req_cover
        and def_cover_title
        and req_title == def_cover_title
        and def_title
        and def_title != def_cover_title
    ):
        store.log(
            f"[publish] 检测到旧版前端缓存将封面标题({req_title})误传为发布标题，已自动恢复为主标题({def_title})"
        )
        req_title = def_title
    final_title = (req_title or def_title).strip()
    final_cover_title = (req_cover or def_cover_title or final_title).strip()
    final_desc = (desc or def_desc or final_title).strip()
    final_tags = [t.strip() for t in (tags if tags is not None else def_tags) if t.strip()]
    tags_csv = ",".join(final_tags)
    rid = run_id or f"pub_{uuid.uuid4().hex[:6]}"
    mode_str = "public" if is_public else "draft"

    synced_covers = sync_platform_covers_for_target(
        manifest=manifest,
        store=store,
        cfg=load_editorial_config(),
        target=target,
        platforms=platforms,
        cover_title_override=final_cover_title,
    )
    if synced_covers:
        store.log(f"[publish] 已预生成 ClipCut 平台封面: {', '.join(p.name for p in synced_covers)}")

    paths = _get_paths()
    caff_prefix = ["/usr/bin/caffeinate", "-i"] if Path("/usr/bin/caffeinate").exists() else []
    selected_platforms = list(dict.fromkeys(p for p in platforms if p in paths))
    if not selected_platforms:
        return store.load()

    started_at = utc_now_iso()

    def _init_running_records(m: ProjectManifest) -> None:
        _, _, _, _, _, target_recs = _default_target_metadata(m, target)
        for plat in selected_platforms:
            label = PLATFORM_LABELS.get(plat, plat)
            _upsert_record(
                target_recs,
                PublishRecord(
                    platform=plat,  # type: ignore[arg-type]
                    status="running",
                    mode=mode_str,  # type: ignore[arg-type]
                    title=final_title,
                    started_at=started_at,
                    updated_at=started_at,
                    progress_pct=0.0,
                    progress_stage="准备中",
                    message=f"正在准备上传到{label}...",
                ),
            )

    store.update_manifest(_init_running_records)
    store.log(
        f"[publish] 并行发布 {target} -> {', '.join(PLATFORM_LABELS.get(p, p) for p in selected_platforms)} ({vpath.name})"
    )

    try:
        publish_timeout_s = float(os.environ.get("CLIPCUT_PUBLISH_TIMEOUT_SECONDS", "1800"))
    except ValueError:
        publish_timeout_s = 1800.0

    def _publish_one_platform(plat: str) -> None:
        pinfo = paths[plat]
        py = pinfo.get("python")
        raw_script = pinfo.get("script")
        script = Path(raw_script) if raw_script else None
        label = PLATFORM_LABELS.get(plat, plat)

        if not py or script is None or not script.is_file():
            err_msg = f"缺少{label}上传脚本或 Python 环境 ({script or '未在 config/local.yaml 配置'})"
            _update_target_record(
                store,
                target,
                PublishRecord(
                    platform=plat,  # type: ignore[arg-type]
                    status="error",
                    mode=mode_str,  # type: ignore[arg-type]
                    title=final_title,
                    started_at=started_at,
                    updated_at=utc_now_iso(),
                    progress_pct=None,
                    progress_stage="",
                    error=err_msg,
                    message=err_msg,
                ),
            )
            store.append_receipt(
                run_id=rid,
                stage="publish",
                actor="system",
                action=f"publish_{plat}",
                summary=err_msg,
                error=err_msg,
            )
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if plat == "bilibili":
            cmd = [
                *caff_prefix,
                py,
                "-u",
                str(script),
                str(vpath),
                final_title[:80],
                "--no_auto_title",
                "--cover-title",
                final_cover_title,
                "--desc",
                final_desc,
                "--tags",
                tags_csv or "人工智能,AI",
            ]
            if tid:
                cmd.extend(["--tid", str(tid)])
            if is_public:
                cmd.append("--public")
        elif plat == "wechat":
            if pinfo.get("profile_dir") is not None:
                env["WECHAT_BROWSER_DATA_DIR"] = str(pinfo["profile_dir"])
            effective_col = (
                collection
                if collection is not None
                else (manifest.metadata.original_theme_zh or manifest.metadata.original_title or "")
            ).strip()[:10]
            cmd = [
                *caff_prefix,
                py,
                "-u",
                str(script),
                str(vpath),
                final_title[:30],
                "--no_auto_title",
                "--cover-title",
                final_cover_title,
                "--desc",
                final_desc,
                "--tags",
                tags_csv or "人工智能,AI",
                "--headless",
            ]
            if effective_col:
                cmd.extend(["--collection", effective_col])
            if is_public:
                cmd.append("--public")
        else:
            # Xiaohongshu automatically uploads same-stem .jpg cover in output directory
            tag_suffix = " ".join(f"#{t}" for t in final_tags[:5] if f"#{t}" not in final_desc)
            xhs_desc = f"{final_desc}\n{tag_suffix}".strip() if tag_suffix else final_desc
            cmd = [
                *caff_prefix,
                py,
                "-u",
                str(script),
                str(vpath),
                final_title[:20],
                xhs_desc,
                "1" if is_public else "0",
            ]

        store.log(f"[publish/{plat}] 开始发布 {target} -> {label} ({vpath.name})")
        state: dict[str, Any] = {"pct": 0.0, "stage": "准备中", "last_save": 0.0}

        def _on_line(line: str) -> None:
            parsed = _parse_publish_progress(plat, line, state["pct"], state["stage"])
            important = any(
                k in line
                for k in ("错误", "失败", "异常", "超时", "❌", "⚠️", "BV", "profile", "锁", "重试")
            )
            if parsed is None:
                if important:
                    store.log(f"[publish/{plat}] {line[:220]}")
                return

            pct, stage, detail_msg = parsed
            stage_changed = stage != state["stage"]
            prev_pct = state["pct"]
            pct_moved = pct is not None and (prev_pct is None or pct != prev_pct)
            now = time.monotonic()

            if stage_changed or important:
                pct_tag = f"{pct:.0f}% " if pct is not None else ""
                store.log(f"[publish/{plat}] {pct_tag}[{stage}] {detail_msg[:180]}")

            should_persist = stage_changed or (
                pct_moved
                and (
                    (now - state["last_save"]) >= 0.8
                    or prev_pct is None
                    or (pct is not None and (pct - prev_pct) >= 5.0)
                )
            )
            if not should_persist:
                return

            state["pct"] = pct
            state["stage"] = stage
            state["last_save"] = now
            _update_target_record(
                store,
                target,
                PublishRecord(
                    platform=plat,  # type: ignore[arg-type]
                    status="running",
                    mode=mode_str,  # type: ignore[arg-type]
                    title=final_title,
                    started_at=started_at,
                    updated_at=utc_now_iso(),
                    progress_pct=pct,
                    progress_stage=stage,
                    message=detail_msg[:200],
                ),
            )

        try:
            rc, out_lines = _run_publish_process(cmd, env, publish_timeout_s, _on_line)
            combined = "\n".join(out_lines).strip()
            tail = out_lines[-8:] if out_lines else []
            if rc == 0:
                bv_m = re.search(r"(BV[0-9A-Za-z]{10})", combined) if plat == "bilibili" else None
                url_id = f"https://www.bilibili.com/video/{bv_m.group(1)}" if bv_m else ""
                done_status = "published" if is_public else "draft"
                summary_msg = f"{label}{'已发布' if is_public else '已存草稿/仅自己可见'}" + (
                    f" ({bv_m.group(1)})" if bv_m else ""
                )
                if plat == "wechat" and (
                    "分享卡片可能仍是微信自动封面" in combined
                    or "分享卡片封面未确认生效" in combined
                ):
                    summary_msg += "（⚠️ 分享卡片封面未确认）"
                _update_target_record(
                    store,
                    target,
                    PublishRecord(
                        platform=plat,  # type: ignore[arg-type]
                        status=done_status,  # type: ignore[arg-type]
                        mode=mode_str,  # type: ignore[arg-type]
                        title=final_title,
                        url_or_id=url_id,
                        started_at=started_at,
                        updated_at=utc_now_iso(),
                        progress_pct=100.0,
                        progress_stage="完成",
                        message=summary_msg,
                    ),
                )
                store.append_receipt(
                    run_id=rid,
                    stage="publish",
                    actor="human",
                    action=f"publish_{plat}",
                    summary=summary_msg,
                    details={
                        "target": target,
                        "file": vpath.name,
                        "title": final_title,
                        "cover_title": final_cover_title,
                        "url_or_id": url_id,
                        "tail": tail,
                    },
                )
            else:
                err_tail = " | ".join(tail[-3:]) or f"退出码 {rc}"
                _update_target_record(
                    store,
                    target,
                    PublishRecord(
                        platform=plat,  # type: ignore[arg-type]
                        status="error",
                        mode=mode_str,  # type: ignore[arg-type]
                        title=final_title,
                        started_at=started_at,
                        updated_at=utc_now_iso(),
                        progress_pct=None,
                        progress_stage="",
                        message=f"{label}发布失败",
                        error=err_tail[:300],
                    ),
                )
                store.append_receipt(
                    run_id=rid,
                    stage="publish",
                    actor="human",
                    action=f"publish_{plat}",
                    summary=f"{label}发布失败",
                    error=err_tail[:400],
                )
        except Exception as exc:
            _update_target_record(
                store,
                target,
                PublishRecord(
                    platform=plat,  # type: ignore[arg-type]
                    status="error",
                    mode=mode_str,  # type: ignore[arg-type]
                    title=final_title,
                    started_at=started_at,
                    updated_at=utc_now_iso(),
                    progress_pct=None,
                    progress_stage="",
                    message=f"{label}发布异常",
                    error=str(exc)[:300],
                ),
            )
            store.append_receipt(
                run_id=rid,
                stage="publish",
                actor="system",
                action=f"publish_{plat}",
                summary=f"{label}发布异常",
                error=str(exc),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_platforms)) as executor:
        futures = {plat: executor.submit(_publish_one_platform, plat) for plat in selected_platforms}
        for fut in futures.values():
            fut.result()

    return store.load()
