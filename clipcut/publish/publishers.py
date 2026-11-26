from __future__ import annotations

import codecs
import concurrent.futures
from datetime import datetime, timezone
import json
import os
import re
import select
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..cancel import (
    JobCancelToken,
    JobCancelledError,
    get_current_cancel_token,
    use_cancel_token,
)
from ..config import load_editorial_config, load_local_config
from ..models import ProjectManifest, PublishRecord, utc_now_iso
from ..pipeline.verify_and_plan import (
    build_default_clean_cut_description,
    build_default_compilation_description,
    build_default_remix_description,
)
from ..receipts import ProjectStore
from ..render.renderer import sync_platform_covers_for_target

PLATFORM_LABELS = {
    "wechat": "微信视频号",
    "bilibili": "B站",
    "xiaohongshu": "小红书",
}

_UNCONFIGURED_HINT = "未配置本地上传脚本（请在 config/local.yaml 或环境变量中配置）"
_ORIG_SUBPROCESS_RUN = subprocess.run
_LOGIN_PROCS: dict[str, dict[str, Any]] = {}
_LOGIN_LOCK = threading.Lock()
_PLATFORM_LOCKS: dict[str, threading.Lock] = {
    "wechat": threading.Lock(),
    "xiaohongshu": threading.Lock(),
}


class _PublishTimeoutError(Exception):
    """Timeout raised by _run_publish_process retaining captured log lines."""

    def __init__(self, timeout_s: float, lines: list[str] | None = None) -> None:
        self.timeout_s = float(timeout_s)
        self.lines = list(lines or [])
        super().__init__(f"上传超时（超过 {self.timeout_s:.0f} 秒）")


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


def _auth_mtime_iso(path: Path | None, *, is_wechat_profile: bool = False) -> str:
    """Return latest credential/cookie mtime in UTC ISO-8601 format."""
    if path is None:
        return ""
    candidates: list[Path] = []
    if is_wechat_profile:
        candidates.extend(
            [
                path / "Default" / "Cookies",
                path / "Default" / "Network" / "Cookies",
                path,
            ]
        )
    else:
        candidates.append(path)
    for cand in candidates:
        try:
            if cand.exists():
                return datetime.fromtimestamp(cand.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            continue
    return ""


def _login_proc_status(platform: str) -> tuple[bool, int | None, str]:
    """Return (login_running, returncode, status_message) for tracked login window."""
    label = PLATFORM_LABELS.get(platform, platform)
    with _LOGIN_LOCK:
        entry = _LOGIN_PROCS.get(platform)
        if not entry:
            return False, None, ""
        proc = entry.get("proc")
    if proc is None:
        return False, None, ""
    rc = proc.poll()
    if rc is None:
        return True, None, f"正在等待{label}扫码登录窗口完成…"
    if rc == 0:
        return False, 0, f"{label}扫码登录已完成，可点击「复位状态」并重试"
    return False, int(rc), f"{label}登录窗口已退出 (code={rc})"


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
    wx_running, wx_rc, wx_login_msg = _login_proc_status("wechat")
    out["wechat"] = {
        "platform": "wechat",
        "label": "微信视频号",
        "logged_in": wx_logged,
        "python_ready": bool(wx["python"]) and wx_script_ready,
        "script": str(wx_script) if wx_script else "",
        "hint": wx_hint,
        "login_running": wx_running,
        "login_returncode": wx_rc,
        "login_updated_at": _auth_mtime_iso(wx_prof, is_wechat_profile=True),
        "login_message": wx_login_msg,
    }

    # 2. Bilibili
    bi = paths["bilibili"]
    bi_script = bi.get("script")
    bi_cookies = bi.get("cookies_file")
    bi_script_ready = bool(bi_script and bi_script.is_file())
    bi_logged, bi_hint = _check_bili_cookies(bi_cookies)
    if not bi_script_ready:
        bi_hint = _UNCONFIGURED_HINT
    bi_running, bi_rc, bi_login_msg = _login_proc_status("bilibili")
    out["bilibili"] = {
        "platform": "bilibili",
        "label": "B站",
        "logged_in": bi_logged,
        "python_ready": bool(bi["python"]) and bi_script_ready,
        "script": str(bi_script) if bi_script else "",
        "hint": bi_hint,
        "login_running": bi_running,
        "login_returncode": bi_rc,
        "login_updated_at": _auth_mtime_iso(bi_cookies),
        "login_message": bi_login_msg,
    }

    # 3. Xiaohongshu
    xhs = paths["xiaohongshu"]
    xhs_script = xhs.get("script")
    xhs_cookies = xhs.get("cookies_file")
    xhs_script_ready = bool(xhs_script and xhs_script.is_file())
    xhs_logged, xhs_hint = _check_xhs_cookies(xhs_cookies)
    if not xhs_script_ready:
        xhs_hint = _UNCONFIGURED_HINT
    xhs_running, xhs_rc, xhs_login_msg = _login_proc_status("xiaohongshu")
    out["xiaohongshu"] = {
        "platform": "xiaohongshu",
        "label": "小红书",
        "logged_in": xhs_logged,
        "python_ready": bool(xhs["python"]) and xhs_script_ready,
        "script": str(xhs_script) if xhs_script else "",
        "hint": xhs_hint,
        "login_running": xhs_running,
        "login_returncode": xhs_rc,
        "login_updated_at": _auth_mtime_iso(xhs_cookies),
        "login_message": xhs_login_msg,
    }
    return out


def launch_platform_login(platform: str) -> dict[str, Any]:
    """Spawn the interactive QR login helper in a background process (non-blocking)."""
    paths = _get_paths()
    if platform not in paths:
        raise ValueError(f"不支持的平台: {platform}")
    label = PLATFORM_LABELS.get(platform, platform)
    with _LOGIN_LOCK:
        existing = _LOGIN_PROCS.get(platform)
        existing_proc = existing.get("proc") if existing else None
        if existing_proc is not None and existing_proc.poll() is None:
            return {
                "started": True,
                "already_running": True,
                "platform": platform,
                "pid": existing_proc.pid,
                "message": f"{label}扫码登录窗口已在运行，请在弹出窗口完成扫码后关闭",
            }

    info = paths[platform]
    py = info.get("python")
    if not py:
        raise RuntimeError(f"未找到 {label} 所需的 Python 运行环境")

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

    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    with _LOGIN_LOCK:
        _LOGIN_PROCS[platform] = {"proc": proc, "started_at": utc_now_iso()}
    return {
        "started": True,
        "already_running": False,
        "platform": platform,
        "pid": proc.pid,
        "message": f"已唤起 {label} 扫码登录窗口，请在弹出窗口完成扫码；关闭窗口后将自动刷新状态",
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
        m_wait = re.search(r"B站接口仍在等待响应[（(](\d+)\s*分钟", clean)
        if m_wait:
            mins = m_wait.group(1)
            return floor, f"等待{mins}m", clean.lstrip("> ")
        if "分块上传失败" in clean or "切换 B站上传线路" in clean or "切换线路" in clean:
            return floor, "重试分块", clean.lstrip("> ")
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
        if "正在设置分享封面" in clean or "正在设置分享卡片封面" in clean:
            return max(floor, 88.0), "设分享封面", clean.lstrip("> ")
        if "保存中转草稿" in clean:
            return max(floor, 90.0), "中转存草稿", clean.lstrip("> ")
        if "正在设置个人主页封面" in clean:
            return max(floor, 92.0), "设主页封面", clean.lstrip("> ")
        if "封面与分享卡片生成完成" in clean or "封面编辑器已可编辑" in clean:
            return max(floor, 88.0), "设封面", clean.lstrip("> ")
        if "封面未生效，关闭弹层等待" in clean:
            return max(floor, 89.0), "重试封面", clean.lstrip("> ")
        if "横竖双封面均已回读确认" in clean:
            return max(floor, 94.0), "封面完成", clean.lstrip("> ")
        if any(k in clean for k in ("点击存草稿按钮", "点击发表按钮", "点击发布按钮")):
            if prev_stage in ("设分享封面", "中转存草稿"):
                return max(floor, 90.0), "中转存草稿", "正在保存中转草稿（分享卡片已就绪，待设主页封面）..."
            return max(floor, 96.0), "提交中", clean.lstrip("> ")
        if "等待发布完成" in clean or "正在到草稿箱核验" in clean:
            if prev_stage in ("设分享封面", "中转存草稿"):
                return max(floor, 91.0), "中转核验", "正在核验中转草稿并准备设置个人主页封面..."
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


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Terminate a subprocess and its entire process group safely."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=0.8)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass


def _run_publish_process(
    cmd: list[str],
    env: dict[str, str],
    timeout_s: float,
    on_line: Callable[[str], None],
) -> tuple[int, list[str]]:
    """Run uploader subprocess while streaming both CR (\r) and LF (\n) progress lines."""
    cancel_token = get_current_cancel_token()
    if cancel_token is not None:
        cancel_token.check("发布任务已手动中止")

    if subprocess.run is not _ORIG_SUBPROCESS_RUN:
        try:
            res = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise _PublishTimeoutError(timeout_s, []) from exc
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
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    if cancel_token is not None:
        cancel_token.register_proc(proc)
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
            if cancel_token is not None and cancel_token.is_cancelled:
                _kill_process_tree(proc)
                raise JobCancelledError("发布任务已手动中止")
            elapsed = time.monotonic() - start
            remaining = timeout_s - elapsed
            if remaining <= 0:
                _kill_process_tree(proc)
                raise _PublishTimeoutError(timeout_s, lines)

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
        if cancel_token is not None and cancel_token.is_cancelled:
            _kill_process_tree(proc)
            raise JobCancelledError("发布任务已手动中止")
        rc = proc.wait(timeout=10)
    finally:
        if cancel_token is not None:
            cancel_token.unregister_proc(proc)
        try:
            proc.stdout.close()
        except Exception:
            pass
    return rc, lines


def _clean_diagnostic_tail(lines: list[str], max_lines: int = 3, max_chars: int = 240) -> str:
    """Filter tracebacks and raw CLI paths to keep only readable error lines."""
    cleaned: list[str] = []
    for raw in lines or []:
        s = re.sub(r"^[>\s]+", "", str(raw or "")).strip()
        if not s:
            continue
        if s.startswith("Traceback (most recent call last):"):
            continue
        if s.startswith('File "') and ", line " in s:
            continue
        if "/usr/bin/caffeinate" in s:
            continue
        if re.search(r"python[0-9.]*\s+-u\s+/.+--login", s) or s.endswith(".py --login"):
            continue
        if s.startswith("Command '[") and "timed out" in s:
            continue
        cleaned.append(s)
    if not cleaned:
        return ""
    joined = " | ".join(cleaned[-max_lines:])
    return joined[:max_chars]


def _diagnose_publish_failure(
    plat: str,
    *,
    rc: int | None = None,
    exc: Exception | None = None,
    lines: list[str] | None = None,
    last_stage: str = "",
    last_pct: float | None = None,
    timeout_s: float | None = None,
) -> tuple[str, str, str, str]:
    """Build structured (error_code, summary_message, error_detail, solution) for a failed publish."""
    label = PLATFORM_LABELS.get(plat, plat)
    eff_lines = list(lines or [])
    if not eff_lines and hasattr(exc, "lines"):
        eff_lines = list(getattr(exc, "lines") or [])
    tail = _clean_diagnostic_tail(eff_lines)
    combined = "\n".join(eff_lines[-20:])
    stage_desc = f"{last_stage} {last_pct:.1f}%" if (last_pct is not None and last_stage) else (last_stage or "准备中")

    if isinstance(exc, JobCancelledError):
        return (
            "cancelled",
            f"{label}上传已中止",
            f"已手动中止{label}上传（中止前阶段：{stage_desc}）",
            "可点击「复位状态」清除当前提示，或直接重新发起上传。",
        )

    if isinstance(exc, (_PublishTimeoutError, subprocess.TimeoutExpired)):
        raw_to = timeout_s or getattr(exc, "timeout_s", None) or getattr(exc, "timeout", None) or 1800.0
        mins = max(1, round(float(raw_to) / 60.0))
        detail = f"{label}上传超过 {mins} 分钟未完成（卡在「{stage_desc}」）" + (
            f"；最后日志：{tail}" if tail else ""
        )
        if plat == "bilibili":
            solution = (
                "1) 检查网络或临时关闭/分流 VPN 代理（B站 UPOS 走国内直连，代理易导致分块上传无限重试卡死）；"
                "2) 点击「复位状态」后单独重试 B站上传；"
                "3) 若多次超时，点「重扫码」刷新 B站 Cookie，或点击「打开输出目录」到 B站创作中心网页端手动投稿。"
            )
        elif plat == "wechat":
            solution = (
                "1) 先到视频号助手草稿箱确认视频是否已上传成功，避免重复提交；"
                "2) 检查直连网络并关闭可能占用 profile 的微信登录窗口；"
                "3) 点击「复位状态」后重试上传。"
            )
        else:
            solution = (
                "1) 先到小红书创作服务平台草稿箱确认是否已保存；"
                "2) 检查本地 Chrome 登录态与网络；"
                "3) 点击「复位状态」后重试上传。"
            )
        return ("timeout", f"{label}上传超时", detail[:360], solution)

    if plat == "wechat":
        if any(k in combined for k in ("登录态不可用", "需要扫码登录", "重新登录失败")):
            return (
                "auth_expired",
                "微信视频号登录态失效",
                "微信视频号登录态已失效，无头上传已停止",
                "1) 点击「重扫码」在弹出浏览器完成扫码登录并等待窗口自动关闭；"
                "2) 登录完成后点击「复位状态」清除失败标记，再重新发起上传。",
            )
        if any(
            k in combined
            for k in ("发布页加载失败", "上传入口未出现", "找不到视频上传入口", "ERR_ABORTED", "草稿箱读取失败")
        ):
            detail = "微信视频号发布页未加载出上传入口（页面加载异常或登录态部分失效）" + (
                f"；详情：{tail}" if tail else ""
            )
            return (
                "page_load_failed",
                "微信视频号发布页加载失败",
                detail[:360],
                "1) 确认未开启代理拦截 channels.weixin.qq.com，且没有其他视频号浏览器窗口占用 profile；"
                "2) 点击「重扫码」在可见窗口确认能进入视频号发布页（如遇风控或账号切换弹窗请先手动处理）；"
                "3) 完成后点击「复位状态」并重试上传。",
            )
        fallback = tail or (str(exc).strip() if exc else "") or f"退出码 {rc}"
        return (
            "upload_failed",
            "微信视频号发布失败",
            f"微信视频号上传中断（阶段：{stage_desc}）：{fallback}"[:360],
            "1) 点击「重扫码」确认视频号创作者平台可正常打开发布页；"
            "2) 关闭残留浏览器窗口后点击「复位状态」并重试上传。",
        )

    if plat == "bilibili":
        if any(k in combined for k in ("cookies.json", "-101", "未登录", "SESSDATA", "412", "Credential")):
            return (
                "auth_expired",
                "B站登录态失效或触发风控",
                f"B站鉴权或风控拦截：{tail or (str(exc).strip() if exc else '') or f'退出码 {rc}'}"[:360],
                "1) 点击「重扫码」重新登录 B站刷新 Cookie；"
                "2) 若触发 412 风控，等待 5-10 分钟或切回国内直连网络；"
                "3) 点击「复位状态」后重试上传。",
            )
        if any(
            k in combined
            for k in ("ConnectError", "Timeout", "TimeoutError", "ServerDisconnected", "分块", "线路", "bvid", "无进度", "网络抖动")
        ):
            return (
                "network",
                "B站上传网络或线路异常",
                f"B站上传中断（阶段：{stage_desc}）：{tail or (str(exc).strip() if exc else '') or f'退出码 {rc}'}"[:360],
                "1) 检查网络并关闭对 bilibili.com / bilivideo.com 的代理；"
                "2) 点击「复位状态」后单独重试 B站上传；"
                "3) 若持续失败，可点「打开输出目录」在 B站创作中心网页端手动投稿。",
            )
        fallback = tail or (str(exc).strip() if exc else "") or f"退出码 {rc}"
        return (
            "upload_failed",
            "B站发布失败",
            f"B站上传异常（阶段：{stage_desc}）：{fallback}"[:360],
            "1) 点击「复位状态」后单独重试 B站上传；"
            "2) 如仍失败，尝试「重扫码」更新 Cookie 或到 B站创作中心网页端手动投稿。",
        )

    # xiaohongshu or other
    if any(k in combined for k in ("cookies", "登录", "扫码")):
        return (
            "auth_expired",
            f"{label}登录态失效",
            f"{label}需要重新扫码登录：{tail or (str(exc).strip() if exc else '') or f'退出码 {rc}'}"[:360],
            f"1) 点击「重扫码」更新{label}登录态；2) 点击「复位状态」后重试上传。",
        )
    fallback = tail or (str(exc).strip() if exc else "") or f"退出码 {rc}"
    return (
        "upload_failed",
        f"{label}发布失败",
        f"{label}上传异常（阶段：{stage_desc}）：{fallback}"[:360],
        f"1) 检查{label}创作者平台页面与本地登录态；2) 点击「复位状态」后重试上传。",
    )


def _default_target_metadata(manifest: ProjectManifest, target: str) -> tuple[Path | None, str, str, str, list[str], list[PublishRecord]]:
    if target == "remix":
        rmx = manifest.remix
        vpath = Path(rmx.output_file) if rmx.output_file else None
        title = (rmx.title or manifest.metadata.original_theme_zh or rmx.cover_title or "高光混剪").strip()
        cover_title = (rmx.cover_title or title).strip()
        desc = build_default_remix_description(manifest)
        tags = list(dict.fromkeys(manifest.primary_topics + ["AI", "深度访谈"]))[:6]
        return vpath, title, cover_title, desc, tags, rmx.publish_records

    if target == "compilation":
        cmp = manifest.compilation
        vpath = Path(cmp.output_file) if cmp.output_file else None
        title = (cmp.title or manifest.metadata.original_theme_zh or cmp.cover_title or "切片精选集").strip()
        cover_title = (cmp.cover_title or title).strip()
        desc = build_default_compilation_description(manifest)
        tags = list(dict.fromkeys(manifest.primary_topics + ["AI", "深度访谈", "合集"]))[:6]
        return vpath, title, cover_title, desc, tags, cmp.publish_records

    if target == "clean_cut":
        cc = manifest.clean_cut
        vpath = Path(cc.output_file) if cc.output_file else None
        fallback_t = f"{manifest.metadata.original_theme_zh or manifest.metadata.original_title or '全片'}（净版）"
        title = (cc.title or cc.cover_title or fallback_t).strip()
        cover_title = (cc.cover_title or title).strip()
        desc = build_default_clean_cut_description(manifest)
        tags = list(dict.fromkeys(manifest.primary_topics + ["AI", "深度访谈", "全片"]))[:6]
        return vpath, title, cover_title, desc, tags, cc.publish_records

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
    cancel_token: JobCancelToken | None = None,
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
        with use_cancel_token(cancel_token):
            pinfo = paths[plat]
            py = pinfo.get("python")
            raw_script = pinfo.get("script")
            script = Path(raw_script) if raw_script else None
            label = PLATFORM_LABELS.get(plat, plat)

            if not py or script is None or not script.is_file():
                err_msg = f"缺少{label}上传脚本或 Python 环境 ({script or '未在 config/local.yaml 配置'})"
                sol_msg = f"请在 config/local.yaml 配置正确的 {label} Python 解释器与上传脚本路径后重启 ClipCut。"
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
                        error_code="script_missing",
                        solution=sol_msg,
                        message=f"缺少{label}上传脚本",
                    ),
                )
                store.log(f"[publish/{plat}] {err_msg} | 建议: {sol_msg}")
                store.append_receipt(
                    run_id=rid,
                    stage="publish",
                    actor="system",
                    action=f"publish_{plat}",
                    summary=err_msg,
                    error=err_msg,
                    details={"error_code": "script_missing", "solution": sol_msg},
                )
                return

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            col_arg = (collection or "").strip()
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
                if col_arg:
                    cmd.extend(["--season_name", col_arg])
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
                if col_arg:
                    cmd.extend(["0", col_arg])

            plat_lock = _PLATFORM_LOCKS.get(plat)
            lock_acquired = False
            try:
                if plat_lock is not None:
                    lock_acquired = plat_lock.acquire(blocking=False)
                    if not lock_acquired:
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
                                progress_pct=0.0,
                                progress_stage="排队中",
                                message=f"等待另一条{label}任务完成后自动开始...",
                            ),
                        )
                        store.log(f"[publish/{plat}] 等待另一条{label}任务完成后自动开始 ({target})")
                        while not lock_acquired:
                            if cancel_token is not None:
                                cancel_token.check("发布任务已手动中止")
                            lock_acquired = plat_lock.acquire(timeout=0.5)

                store.log(f"[publish/{plat}] 开始发布 {target} -> {label} ({vpath.name})")
                state: dict[str, Any] = {"pct": 0.0, "stage": "准备中", "last_save": 0.0}

                def _on_line(line: str) -> None:
                    parsed = _parse_publish_progress(plat, line, state["pct"], state["stage"])
                    important = any(
                        k in line
                        for k in ("错误", "失败", "异常", "超时", "❌", "⚠️", "BV", "profile", "锁", "重试", "等待响应", "分块", "线路")
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
                        error_code, summary_msg, err_detail, sol_msg = _diagnose_publish_failure(
                            plat,
                            rc=rc,
                            lines=out_lines,
                            last_stage=state["stage"],
                            last_pct=state["pct"],
                            timeout_s=publish_timeout_s,
                        )
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
                                message=summary_msg,
                                error=err_detail[:360],
                                error_code=error_code,
                                solution=sol_msg,
                            ),
                        )
                        store.log(f"[publish/{plat}] {err_detail} | 建议: {sol_msg}")
                        store.append_receipt(
                            run_id=rid,
                            stage="publish",
                            actor="human",
                            action=f"publish_{plat}",
                            summary=summary_msg,
                            error=err_detail[:400],
                            details={
                                "error_code": error_code,
                                "solution": sol_msg,
                                "tail": tail,
                            },
                        )
                except Exception as exc:
                    captured_lines = list(getattr(exc, "lines", []) or [])
                    error_code, summary_msg, err_detail, sol_msg = _diagnose_publish_failure(
                        plat,
                        exc=exc,
                        lines=captured_lines,
                        last_stage=state["stage"],
                        last_pct=state["pct"],
                        timeout_s=publish_timeout_s,
                    )
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
                            message=summary_msg,
                            error=err_detail[:360],
                            error_code=error_code,
                            solution=sol_msg,
                        ),
                    )
                    store.log(f"[publish/{plat}] {err_detail} | 建议: {sol_msg}")
                    store.append_receipt(
                        run_id=rid,
                        stage="publish",
                        actor="system",
                        action=f"publish_{plat}",
                        summary=summary_msg,
                        error=err_detail[:400],
                        details={
                            "error_code": error_code,
                            "solution": sol_msg,
                            "tail": captured_lines[-8:],
                        },
                    )
            except JobCancelledError as exc:
                error_code, summary_msg, err_detail, sol_msg = _diagnose_publish_failure(
                    plat,
                    exc=exc,
                    last_stage="排队中",
                    last_pct=0.0,
                    timeout_s=publish_timeout_s,
                )
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
                        message=summary_msg,
                        error=err_detail[:360],
                        error_code=error_code,
                        solution=sol_msg,
                    ),
                )
                store.log(f"[publish/{plat}] {err_detail}")
            finally:
                if lock_acquired and plat_lock is not None:
                    plat_lock.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_platforms)) as executor:
        futures = {plat: executor.submit(_publish_one_platform, plat) for plat in selected_platforms}
        for fut in futures.values():
            fut.result()

    if cancel_token is not None and cancel_token.is_cancelled:
        raise JobCancelledError("发布任务已手动中止")

    return store.load()


def publish_batch_targets(
    *,
    store: ProjectStore,
    targets: list[str],
    platforms: list[str],
    is_public: bool = False,
    collection: str | None = None,
    skip_already_published: bool = True,
    tid: int | None = None,
    extra_tags: list[str] | None = None,
    run_id: str | None = None,
    cancel_token: JobCancelToken | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sequentially publish a batch of standalone clips using each clip's metadata and shared collection."""
    rid = run_id or f"pubbatch_{uuid.uuid4().hex[:6]}"
    clean_platforms = list(dict.fromkeys(p for p in platforms if p in PLATFORM_LABELS))
    col_str = (collection or "").strip()
    clean_extra_tags = [str(t).strip() for t in (extra_tags or []) if str(t).strip()]

    completed_targets: list[str] = []
    failed_targets: list[str] = []
    skipped_targets: list[str] = []

    def _emit_progress(current_target: str | None) -> None:
        if on_progress is None:
            return
        on_progress(
            {
                "current_target": current_target,
                "completed_targets": list(completed_targets),
                "failed_targets": list(failed_targets),
                "skipped_targets": list(skipped_targets),
            }
        )

    store.log(
        f"[publish/batch] 开始批量发布 {len(targets)} 条切片 -> {', '.join(PLATFORM_LABELS.get(p, p) for p in clean_platforms)}"
        + (f"（合集：{col_str}）" if col_str else "")
    )

    for idx, tid_str in enumerate(targets):
        if cancel_token is not None:
            cancel_token.check("批量发布任务已手动中止")

        manifest = store.load()
        try:
            vpath, def_title, def_cover_title, def_desc, def_tags, rec_list = _default_target_metadata(manifest, tid_str)
        except ValueError:
            skipped_targets.append(tid_str)
            _emit_progress(None)
            continue

        if vpath is None or not vpath.exists():
            store.log(f"[publish/batch] 跳过 {tid_str}（尚未渲染视频文件）")
            skipped_targets.append(tid_str)
            _emit_progress(None)
            continue

        target_platforms = list(clean_platforms)
        if skip_already_published:
            done_plats = {r.platform for r in rec_list if r.status in ("published", "draft")}
            target_platforms = [p for p in target_platforms if p not in done_plats]
            if not target_platforms:
                store.log(f"[publish/batch] 跳过 {tid_str}（所选平台均已存草稿或已发布）")
                skipped_targets.append(tid_str)
                _emit_progress(None)
                continue

        merged_tags = list(dict.fromkeys(def_tags + clean_extra_tags))[:6] if clean_extra_tags else def_tags
        _emit_progress(tid_str)

        try:
            updated_manifest = publish_to_platforms(
                store=store,
                target=tid_str,
                platforms=target_platforms,
                title=def_title,
                cover_title=def_cover_title,
                desc=def_desc,
                tags=merged_tags,
                is_public=is_public,
                collection=col_str,
                tid=tid,
                run_id=f"{rid}_{idx + 1:02d}",
                ui_schema=2,
                cancel_token=cancel_token,
            )
            _, _, _, _, _, after_recs = _default_target_metadata(updated_manifest, tid_str)
            rec_by_plat = {r.platform: r for r in after_recs}
            if any(rec_by_plat.get(p) and rec_by_plat[p].status == "error" for p in target_platforms):
                failed_targets.append(tid_str)
            else:
                completed_targets.append(tid_str)
        except JobCancelledError:
            raise
        except Exception as exc:
            store.log(f"[publish/batch] 发布 {tid_str} 异常: {exc}")
            failed_targets.append(tid_str)

        _emit_progress(None)

        if idx + 1 < len(targets) and any(p in ("wechat", "xiaohongshu") for p in target_platforms):
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if cancel_token is not None:
                    cancel_token.check("批量发布任务已手动中止")
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    summary = (
        f"批量发布结束：成功 {len(completed_targets)} 支，跳过 {len(skipped_targets)} 支，失败 {len(failed_targets)} 支"
        + (f"（合集：{col_str}）" if col_str else "")
    )
    store.log(f"[publish/batch] {summary}")
    store.append_receipt(
        run_id=rid,
        stage="publish",
        actor="human",
        action="batch_publish_complete",
        summary=summary,
        details={
            "targets": list(targets),
            "platforms": clean_platforms,
            "collection": col_str,
            "completed_targets": completed_targets,
            "failed_targets": failed_targets,
            "skipped_targets": skipped_targets,
        },
    )
    return {
        "completed_targets": completed_targets,
        "failed_targets": failed_targets,
        "skipped_targets": skipped_targets,
        "collection": col_str,
    }


def reset_publish_records(
    store: ProjectStore,
    target: str,
    platform: str | None = None,
    *,
    only_errors: bool = False,
) -> tuple[ProjectManifest, list[str]]:
    """Clear failed/stale (or all) PublishRecord entries for a target so UI state resets cleanly."""
    if platform is not None and platform not in PLATFORM_LABELS:
        raise ValueError(f"不支持的平台: {platform}")
    removed: list[str] = []

    def _mutate(m: ProjectManifest) -> None:
        _, _, _, _, _, rec_list = _default_target_metadata(m, target)
        kept: list[PublishRecord] = []
        for rec in rec_list:
            match_plat = platform is None or rec.platform == platform
            match_state = (not only_errors) or rec.status in ("error", "running")
            if match_plat and match_state:
                removed.append(rec.platform)
            else:
                kept.append(rec)
        rec_list[:] = kept

    updated = store.update_manifest(_mutate)
    if removed:
        labels = ", ".join(PLATFORM_LABELS.get(p, p) for p in removed)
        store.log(f"[publish] 已复位 {target} 发布状态: {labels}")
        store.append_receipt(
            run_id=f"reset_{uuid.uuid4().hex[:6]}",
            stage="publish",
            actor="human",
            action="reset_publish_records",
            summary=f"已复位 {target} 发布状态 ({labels})",
            details={"target": target, "removed": removed, "only_errors": only_errors},
        )
    return updated, removed
