"""一键配置环境：从「刚下载完这个仓库」到「能打开网页」。

这是「setup.bat」真正的实现。和 ``scripts/selftest.py`` 一样，逻辑放在 Python 里
而不是 ``.bat`` 里，原因相同：``.bat`` 不能含非 ASCII 字节（cmd.exe 会按当前代码页
分块解码，中文会把后面的行吃掉，详见 :mod:`fisheries.launcher` 的模块说明），
而这里要输出的全是中文提示。

它做的事（每一步都可单独跳过，重复运行是安全的）
------------------------------------------------
1. 检查 Python 版本（``pyproject.toml`` 要求 >= 3.10）；
2. 建 ``.venv``；
3. 升级 pip / setuptools / wheel；
4. ``pip install -e ".[dev,rag,api]"``；
5. 构建知识库向量库（**向量库不入库**，所以全新克隆必须做一次；``--skip-kb`` 跳过）；
6. 验证 ``import fisheries`` 与接口依赖，并打印下一步。

为什么第 5 步是必需的
--------------------
``knowledge/chroma/`` 在 ``.gitignore`` 里。不建的话接口照样起得来、页面也能开，
只有「知识问答」会返回 503（见 ``src/fisheries/api/deps.py``）。所以这里默认建，
但明确告知会下载约 91 MB 的向量模型，并且允许跳过——用户可能只想先看看页面。

用法：

    python scripts/setup_env.py                  # 一条命令配好
    python scripts/setup_env.py --mirror tsinghua  # 用国内镜像装依赖
    python scripts/setup_env.py --skip-kb        # 先不建向量库
    python scripts/setup_env.py --recreate       # 删掉 .venv 重建

退出码：0 成功；1 有失败步骤。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fisheries import console as _console  # noqa: E402  (必须在 sys.path 处理之后)

VENV_DIR = PROJECT_ROOT / ".venv"
VENV_BIN = "Scripts" if os.name == "nt" else "bin"
VENV_PY = VENV_DIR / VENV_BIN / ("python.exe" if os.name == "nt" else "python")
KB_MANIFEST = PROJECT_ROOT / "knowledge" / "chroma" / "fisheries_manifest.json"

#: 与 pyproject.toml 的 requires-python 保持一致。
MIN_PYTHON = (3, 10)

#: 默认安装的 extras。dev 带 pytest（自检要用），rag 带知识库，api 带接口与前端托管。
DEFAULT_EXTRAS: tuple[str, ...] = ("dev", "rag", "api")

#: 可选的额外 extras（重量级，默认不装）。
EXTENDED_EXTRAS: tuple[str, ...] = ("translate", "vision", "data", "datasets", "train")

#: 国内镜像。空串 = 走 pip 默认索引。
MIRRORS: dict[str, str] = {
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "ustc": "https://mirrors.ustc.edu.cn/pypi/simple",
    "official": "",
}

#: HuggingFace 的国内镜像。``fastembed`` 通过 ``huggingface_hub`` 下模型，
#: 后者认 ``HF_ENDPOINT`` 这个环境变量，所以不用改任何代码就能换源。
#: 实测（本机）：``huggingface.co`` 直连超时，``hf-mirror.com`` 正常。
HF_MIRROR = "https://hf-mirror.com"

_BAR = "  " + "-" * 60
_RULE = "  " + "=" * 60


# --------------------------------------------------------------------------- #
# 一、纯函数（可单测，不碰网络也不碰磁盘）
# --------------------------------------------------------------------------- #

def python_version_ok(version_info: tuple[int, int]) -> bool:
    """``(major, minor)`` 是否满足最低版本要求。"""
    return tuple(version_info) >= MIN_PYTHON


def parse_extras(text: str) -> tuple[str, ...]:
    """把 ``"dev, rag,api"`` 解析成 ``("dev", "rag", "api")``；空串得到空元组。"""
    return tuple(part.strip() for part in text.split(",") if part.strip())


def pip_spec(extras: tuple[str, ...]) -> str:
    """构造给 pip 的安装目标，如 ``".[dev,rag,api]"``。"""
    if not extras:
        return "."
    return ".[" + ",".join(extras) + "]"


def mirror_url(name: str) -> str | None:
    """镜像名 → 索引地址；未知名返回 ``None``（由调用方报错）。"""
    return MIRRORS.get(name)


def pip_install_command(python: Path, extras: tuple[str, ...], index_url: str | None) -> list[str]:
    """拼出 ``pip install`` 的完整命令行（用列表传参，不经过 shell）。"""
    cmd = [str(python), "-m", "pip", "install", "-e", pip_spec(extras)]
    if index_url:
        cmd += ["-i", index_url]
    return cmd


def is_windows_store_python(python: Path) -> bool:
    """Windows 商店版 python 是转发器，建 venv 会失败且报错难懂，先挡掉。"""
    return "windowsapps" in str(python).lower()


# --------------------------------------------------------------------------- #
# 二、执行辅助
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """跑一条外部命令并原样转出它的输出；返回退出码。"""
    print(_BAR)
    print("  $ " + " ".join(cmd))
    print(_BAR)
    # 输出被重定向到文件时 stdout 是块缓冲的，不刷一下会让「步骤标题」出现在
    # 子进程输出的后面（实测：整块横幅跑到最后）。子进程也设 UNBUFFERED，
    # 否则它的进度同样会憋到最后一次性吐出，顺序全乱。
    sys.stdout.flush()
    merged = dict(os.environ)
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    merged["PYTHONUNBUFFERED"] = "1"
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), env=merged).returncode


def _has_venv() -> bool:
    return VENV_PY.is_file()


# --------------------------------------------------------------------------- #
# 三、各步骤
# --------------------------------------------------------------------------- #

def step_python(args: argparse.Namespace) -> bool:
    print("  [1/7] 检查 Python")
    version = sys.version_info
    print(f"        当前解释器：{sys.executable}")
    print(f"        版本：{version.major}.{version.minor}.{version.micro}")

    if not python_version_ok((version.major, version.minor)):
        need = ".".join(str(x) for x in MIN_PYTHON)
        print(f"        [失败] 需要 Python >= {need}，当前是 {version.major}.{version.minor}")
        print("        安装地址：https://www.python.org/downloads/")
        print('        安装时记得勾选 "Add python.exe to PATH"。')
        return False

    if is_windows_store_python(Path(sys.executable)):
        print("        [失败] 这是 Windows 商店版的 python 转发器，建虚拟环境会失败。")
        print("        请从 python.org 装一个正式版，或用 --python 指定解释器路径。")
        return False

    print(f"        {_console.mark(True)} 版本满足要求")
    return True


def step_venv(args: argparse.Namespace) -> bool:
    print("  [2/7] 准备虚拟环境 .venv")

    if _has_venv() and not args.recreate:
        print(f"        {_console.mark(True)} 已存在：{VENV_DIR}（要重建加 --recreate）")
        return True

    if VENV_DIR.exists() and args.recreate:
        print(f"        删除旧环境：{VENV_DIR}")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    interpreter = Path(args.python).expanduser() if args.python else Path(sys.executable)
    if not interpreter.is_file() and shutil.which(str(interpreter)) is None:
        print(f"        [失败] 找不到解释器：{interpreter}")
        return False

    print(f"        用 {interpreter} 创建 {VENV_DIR}")
    rc = _run([str(interpreter), "-m", "venv", str(VENV_DIR)])
    if rc != 0 or not _has_venv():
        print(f"        [失败] 创建虚拟环境失败（退出码 {rc}）")
        return False

    print(f"        {_console.mark(True)} {VENV_PY}")
    return True


def step_pip(args: argparse.Namespace) -> bool:
    print("  [3/7] 升级 pip / setuptools / wheel")
    rc = _run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if rc != 0:
        print(f"        [失败] 升级失败（退出码 {rc}）。若是网络问题，加 --mirror tsinghua 重试。")
        return False
    return True


def step_install(args: argparse.Namespace) -> bool:
    extras = parse_extras(args.extras)
    index_url = mirror_url(args.mirror)
    print(f"  [4/7] 安装项目依赖 {pip_spec(extras)}")

    if args.skip_install:
        print("        （--skip-install，已跳过）")
        return True

    rc = _run(pip_install_command(VENV_PY, extras, index_url))
    if rc != 0:
        print(f"        [失败] 安装失败（退出码 {rc}）")
        print("        常见原因：网络不通或没开镜像。加 --mirror tsinghua 再试。")
        return False
    return True


def step_datasets(args: argparse.Namespace) -> bool:
    print("  [5/7] 导入数据集（可选，最大约 1.4 GB）")
    if args.skip_datasets:
        print("        （--skip-datasets，已跳过）")
        print("        之后补导： .venv\\Scripts\\python.exe scripts\\fetch_datasets.py --fetch all")
        return True

    # 本地已有的会自己跳过（fetch_datasets 先 find 再下载），所以数据齐的机器上这一步秒过。
    cmd = [str(VENV_PY), "scripts/fetch_datasets.py", "--fetch", "all"]
    env = {"FISHERIES_DATASET_MIRROR": args.dataset_mirror} if args.dataset_mirror else None
    if args.dataset_mirror:
        print(f"        镜像源：{args.dataset_mirror}")
    print("        先试 kaggle，失败自动切镜像（镜像包不需要 Kaggle 凭证）。")
    rc = _run(cmd, env=env)

    if rc != 0:
        print()
        print("        [部分失败] 有数据集没导进来。不影响页面与接口先跑起来，")
        print("        M8 评测、病害图像演示会缺数据。三条路任选：")
        print("          1) 配 Kaggle 凭证后重试：")
        print('               .venv\\Scripts\\python.exe -m pip install -e ".[datasets]"')
        print("               凭证放 %USERPROFILE%\\.kaggle\\kaggle.json")
        print("          2) 用国内镜像（不需要凭证）：")
        print("               一键配置环境.bat --dataset-mirror https://你的地址")
        print("          3) 有现成数据就指过去：")
        print("               set FISHERIES_DATA_ROOT=<放数据集的总目录>")
        return False
    return True


def step_knowledge_base(args: argparse.Namespace) -> bool:
    print("  [6/7] 构建知识库向量库")
    if args.skip_kb:
        print("        （--skip-kb，已跳过；知识问答会不可用）")
        print("        之后补建： .venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
        return True

    if KB_MANIFEST.exists() and not args.rebuild_kb:
        print(f"        {_console.mark(True)} 已存在：{KB_MANIFEST.relative_to(PROJECT_ROOT)}（要重建加 --rebuild-kb）")
        return True

    print("        首次构建会下载约 91 MB 的中文向量模型到 models/fastembed/，请稍等。")
    cmd = [str(VENV_PY), "scripts/build_knowledge_base.py"]

    env = {"HF_ENDPOINT": HF_MIRROR} if args.hf_mirror else None
    rc = _run(cmd, env=env)

    if rc != 0 and not args.hf_mirror:
        # huggingface.co 在国内经常直连超时（实测本机就是），而 hf-mirror.com 正常。
        # 与其让用户读一屏 traceback 再去搜，不如直接换源重试一次。
        print()
        print(f"        默认下载源失败，改走镜像 {HF_MIRROR} 重试一次……")
        rc = _run(cmd, env={"HF_ENDPOINT": HF_MIRROR})

    if rc != 0:
        print(f"        [失败] 构建失败（退出码 {rc}）")
        print("        页面与接口不受影响，只是「知识问答」不可用。可先跳过：--skip-kb")
        print("        手动排查： .venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
        print(f"        网络受限时先设 HF_ENDPOINT={HF_MIRROR} 再跑上面那条。")
        return False
    return True


def step_verify(args: argparse.Namespace) -> bool:
    print("  [7/7] 验证安装")
    if args.skip_verify:
        print("        （--skip-verify，已跳过）")
        return True

    # 一行搞定，否则 _run() 把命令原样回显时会拆成好几行，看着像出错。
    # importlib.util 是子模块：只写 import importlib 拿不到 .util，
    # Python 3.13 上实测 AttributeError。
    probe = (
        "import sys, importlib.util; "
        "missing = [m for m in ('fisheries', 'fastapi', 'uvicorn', 'chromadb') "
        "if importlib.util.find_spec(m) is None]; "
        "print(('缺失：' + ', '.join(missing)) if missing else '依赖齐全'); "
        "sys.exit(1 if missing else 0)"
    )
    rc = _run([str(VENV_PY), "-c", probe])
    if rc != 0:
        print(f"        [失败] 验证脚本退出码 {rc}。")
        print("        上面若打印了「缺失：…」，按缺的包名重装；若是 Traceback，")
        print("        那是脚本自身的问题，请把输出一并反馈。")
        return False
    return True


def print_next_steps(
    core_ok: bool, data_ok: bool, kb_ok: bool, verify_ok: bool, args: argparse.Namespace
) -> None:
    print()
    print(_RULE)
    if core_ok and verify_ok and kb_ok:
        print("    环境已就绪。下一步：")
        print()
        print("      1. 双击  启动接口服务.bat")
        print("      2. 浏览器打开横幅里给出的地址（形如 http://127.0.0.1:8000/app/）")
        print()
        print("    可选：")
        print("      · 先跑一遍自检：  双击 运行自检.bat")
        if data_ok:
            print("      · 数据集已导好，M8 评测与病害图像演示都能跑")
        else:
            print("      · 数据集还没导全（可选，约 1.4 GB）。没有 Kaggle 凭证就用镜像：")
            print("          一键配置环境.bat --dataset-mirror https://你的地址")
            print("          .venv\\Scripts\\python.exe scripts\\fetch_datasets.py --status")
        print("      · 模型权重未随仓库分发，没有它照样能用（病害识别走模拟兜底）")
        if args.skip_kb:
            print("      · 知识问答需要向量库：")
            print("          .venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
    elif core_ok and verify_ok and not kb_ok:
        print("    环境基本就绪，只差知识库向量库。")
        print()
        print("      页面和接口现在就能用（双击 启动接口服务.bat），")
        print("      只有「知识问答」会返回 503。补建：")
        print("          .venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
        print(f"      下载仍失败就先设 HF_ENDPOINT={HF_MIRROR} 再跑。")
    else:
        print("    配置未完成。把上面的 [失败] 行连同报错贴出来即可定位。")
        print("    常见处理：加 --mirror tsinghua 换源重试；或 --recreate 重建虚拟环境。")
    print(_RULE)
    print()


# --------------------------------------------------------------------------- #
# 四、入口
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键配置运行环境（建 .venv、装依赖、建向量库、验证）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--extras", default=",".join(DEFAULT_EXTRAS),
                        help=f"要安装的 extras，逗号分隔。可选：{', '.join(EXTENDED_EXTRAS)}")
    parser.add_argument("--mirror", default="official", choices=sorted(MIRRORS),
                        help="PyPI 镜像")
    parser.add_argument("--python", default=None,
                        help="用来创建虚拟环境的解释器路径（默认当前解释器）")
    parser.add_argument("--recreate", action="store_true", help="删掉 .venv 重建")
    parser.add_argument("--rebuild-kb", action="store_true", help="即使向量库已存在也重建")
    parser.add_argument("--hf-mirror", action="store_true",
                        help=f"下向量模型时直接用国内镜像 {HF_MIRROR}")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="跳过数据集导入（最大约 1.4 GB）")
    parser.add_argument("--dataset-mirror", default=None, metavar="URL",
                        help="数据集国内镜像根地址；给了就不需要 Kaggle 凭证")
    parser.add_argument("--skip-kb", action="store_true", help="跳过向量库构建")
    parser.add_argument("--skip-verify", action="store_true", help="跳过最终验证")
    return parser


def main(argv: list[str] | None = None) -> int:
    _console.install()
    parser = build_parser()
    args = parser.parse_args(argv)

    print()
    print(_RULE)
    print("    ProjectFisheries 一键配置环境")
    print(_RULE)
    print(f"    工程目录：{PROJECT_ROOT}")
    print()

    ok = step_python(args)
    if ok:
        print()
        ok = step_venv(args)
    if ok:
        print()
        ok = step_pip(args)
    if ok:
        print()
        ok = step_install(args)

    # 数据集与向量库的失败都不该吞掉后面的验证：依赖装没装好是另一件事，
    # 得单独告诉用户。
    data_ok = False
    if ok:
        print()
        data_ok = step_datasets(args)

    kb_ok = False
    if ok:
        print()
        kb_ok = step_knowledge_base(args)

    verify_ok = False
    if ok:
        print()
        verify_ok = step_verify(args)

    print_next_steps(ok, data_ok, kb_ok, verify_ok, args)
    # 数据集是可选的（缺了只是评测与图像演示缺数据），所以不参与退出码。
    return 0 if (ok and verify_ok and kb_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
