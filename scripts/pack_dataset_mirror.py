"""把本地已有的数据集（和模型权重）打成镜像，供没有 Kaggle 凭证的人下载。

    python scripts/pack_dataset_mirror.py                 # 打包代码真正依赖的两份
    python scripts/pack_dataset_mirror.py --all            # 本地有什么就打什么
    python scripts/pack_dataset_mirror.py --name fish_disease sb_fish_disease
    python scripts/pack_dataset_mirror.py --weights        # 顺带把模型权重也放进去
    python scripts/pack_dataset_mirror.py --skip-datasets --weights   # 只打包权重
    python scripts/pack_dataset_mirror.py --dry-run        # 只报体积，不写文件

产物（默认 ``dist/mirror/``，不入库）：

    dist/mirror/
    ├── fish_disease.zip
    ├── aquaponics_pond.zip
    ├── weights/                    # --weights 才有
    │   ├── yolov8-fish.pt
    │   └── yolov8-fish-lesion-seg.pt
    └── manifest.json       # 每个包的 sha256 / 字节数 / 文件数，以及权重 md5

## 为什么需要它

``fetch_datasets.py --fetch`` 默认走 kagglehub，要 Kaggle 账号 + 凭证；
没有凭证的人只能干看着。镜像包解决这件事：把它上传到任意静态托管
（GitHub Release、阿里云 OSS、腾讯 COS、对象存储、甚至内网文件服务器），
然后设一个环境变量就能装：

    set FISHERIES_DATASET_MIRROR=https://your-host/fish-datasets
    python scripts/fetch_datasets.py --fetch all
    python scripts/fetch_datasets.py --fetch-weights all

``fetch`` 会先试 kaggle，失败自动切镜像，**不需要任何凭证**。sha256 从
``manifest.json`` 里读，所以注册表一行都不用改。

权重不压缩：``fetch-weights`` 直接下载原文件再核 md5/sha256，少一层解压也少一个
出错的地方；官方预训练权重（``yolov8n-*``）本来就能从 ultralytics release 取，
不重复托管。

## 打包是可复现的

zip 成员按路径排序、时间戳固定成 1980-01-01，同一份数据重复打包得到**同一个
sha256**。这样 sha256 才能写进注册表当校验值，而不是每次 CI 都变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认 GBK，装饰符号编不出来会打断脚本。详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as ds  # noqa: E402
from fisheries import weights as wt  # noqa: E402

#: 代码真正依赖的数据集——不带参数时只打这两份，别的都是备选。
DEFAULT_NAMES = ("fish_disease", "aquaponics_pond")

#: zip 成员统一用这个时间戳，保证同一份数据重复打包字节一致。
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)

#: 单文件数据集也按这个名字放进压缩包根目录（与 ``DatasetSpec.entry`` 一致）。


def _rule(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_payload(source: Path, entry: str):
    """列出要写进压缩包的 ``(成员名, 源文件)``。

    ``source`` 是目录 → 目录内容作为成员名；``source`` 是单文件 → 用 ``entry``
    当成员名（单文件数据集的 ``entry`` 本身就是文件名）。
    """
    if source.is_file():
        yield entry, source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            yield path.relative_to(source).as_posix(), path


def pack_one(source: Path, entry: str, archive: Path, *, level: int) -> dict[str, object]:
    """把一个数据集打成 zip，返回它的清单条目。"""
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".part")
    files = 0
    raw_bytes = 0
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=level
    ) as bundle:
        for member, path in iter_payload(source, entry):
            info = zipfile.ZipInfo(member, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0644：解出来权限一致，不受打包机 umask 影响。
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
            files += 1
            raw_bytes += path.stat().st_size
    temporary.replace(archive)
    return {
        "file": archive.name,
        "sha256": _sha256_of(archive),
        "bytes": archive.stat().st_size,
        "uncompressed_bytes": raw_bytes,
        "files": files,
        "entry": entry,
    }


def _human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        value /= 1024
    return f"{value:,.1f} GB"


def write_manifest(
    output: Path,
    entries: dict[str, dict[str, object]],
    weights: dict[str, dict[str, object]] | None = None,
) -> Path:
    """写 ``manifest.json``——下载时 ``fetch`` 就是从这份文件取 sha256 的。"""
    payload: dict[str, object] = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "由 scripts/pack_dataset_mirror.py 生成；上传到静态托管后用 "
                "FISHERIES_DATASET_MIRROR 指向该目录即可",
        "datasets": entries,
    }
    if weights:
        payload["weights"] = weights
    path = output / ds.MIRROR_MANIFEST_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _existing_datasets(output: Path) -> dict[str, dict[str, object]]:
    """读回已有 manifest 里的 ``datasets`` 段。

    ``--weights --skip-datasets`` 只处理权重，但 manifest 是整份覆盖的：
    不把原数据集段带过来，就会把上一次打包的成果从清单里抹掉。
    """
    path = output / ds.MIRROR_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    datasets = payload.get("datasets")
    return datasets if isinstance(datasets, dict) else {}


def pack_weights(
    output: Path, *, dry_run: bool = False
) -> dict[str, dict[str, object]]:
    """把「我们自己训的」权重原样放进 ``<out>/weights/``，返回清单条目。

    官方预训练权重（``upstream`` 非空）跳过：它们本来就能从 ultralytics release
    取，重复托管只是让镜像站多一份会过期的东西。
    """
    table: dict[str, dict[str, object]] = {}
    target = output / wt.MIRROR_SUBDIR
    for name in sorted(wt.WEIGHTS):
        item = wt.spec(name)
        if item.upstream:
            print(f"  · {name:<14} 官方权重，走上游地址，不入镜像")
            continue
        if not item.path.is_file():
            print(f"  · {name:<14} 本地没有，跳过")
            continue
        size = item.path.stat().st_size
        if dry_run:
            print(f"  · {name:<14} {_human(size):>12}  {item.path}")
            continue
        ok, detail = wt.verify(name)
        if not ok:
            print(f"  ✘ {name:<14} 本地文件校验不过，跳过：{detail}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        dest = target / item.filename
        shutil.copyfile(item.path, dest)
        table[name] = {
            "file": f"{wt.MIRROR_SUBDIR}/{item.filename}",
            "md5": wt.md5_of(dest),
            "sha256": wt.sha256_of(dest),
            "bytes": dest.stat().st_size,
            "label": item.label,
            "purpose": item.purpose,
        }
        print(
            f"  ✔ {name:<14} {_human(int(table[name]['bytes'])):>12}"
            f"  md5 {str(table[name]['md5'])[:12]}…  → {dest.relative_to(output).as_posix()}"
        )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pack_dataset_mirror",
        description="把本地数据集打成镜像压缩包（供没有 Kaggle 凭证的人下载）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/pack_dataset_mirror.py --dry-run\n"
            "  python scripts/pack_dataset_mirror.py --all\n"
            "  python scripts/pack_dataset_mirror.py --name fish_disease\n"
        ),
    )
    parser.add_argument("--out", default=str(ds.MIRROR_OUTPUT_DIR), help="输出目录")
    parser.add_argument(
        "--name", nargs="+", metavar="NAME", help="只打这些（默认代码依赖的两份）"
    )
    parser.add_argument("--all", action="store_true", help="本地有哪份就打哪份")
    parser.add_argument("--level", type=int, default=6, help="zip 压缩级别 1-9")
    parser.add_argument("--weights", action="store_true", help="顺带把模型权重放进 <out>/weights/")
    parser.add_argument(
        "--skip-datasets", action="store_true", help="不打包数据集，只处理权重（配合 --weights）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只报体积，不写文件")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出清单")
    args = parser.parse_args(argv)

    if not 1 <= args.level <= 9:
        print("  ✘ --level 只能取 1-9")
        return 2
    if args.skip_datasets and not args.weights:
        print("  ✘ --skip-datasets 要配合 --weights（否则什么也不打）")
        return 2

    if args.all:
        names = [spec.name for spec in ds.list_datasets()]
    elif args.name:
        names = list(args.name)
    else:
        names = list(DEFAULT_NAMES)

    output = Path(args.out)
    entries: dict[str, dict[str, object]] = {}
    weight_entries: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    print()
    _rule("=")
    print("  镜像打包（数据集 + 模型权重）")
    _rule("=")
    print(f"  输出目录  {output}")
    print(f"  数据集     {'（跳过）' if args.skip_datasets else f'{len(names)} 个'}")
    print(f"  权重       {'打包' if args.weights else '（不打包，加 --weights 开启）'}")
    print()

    if not args.skip_datasets:
        print("  —— 数据集 ——")
        for name in names:
            if name not in ds.DATASETS:
                print(f"  ✘ {name:<28} 未注册")
                missing.append(name)
                continue
            spec = ds.DATASETS[name]
            source = ds.find(name)
            if source is None:
                print(f"  · {name:<28} 本地没有，跳过（先 --fetch 再来打包）")
                missing.append(name)
                continue
            if args.dry_run:
                size = (
                    source.stat().st_size
                    if source.is_file()
                    else sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
                )
                print(f"  · {name:<28} {_human(size):>12}  {source}")
                continue
            entry = pack_one(source, spec.entry, output / f"{name}.zip", level=args.level)
            entries[name] = entry
            print(
                f"  ✔ {name:<28} {_human(int(entry['bytes'])):>12}"
                f"  {entry['files']:>7,} 文件  sha256 {str(entry['sha256'])[:12]}…"
            )
        if entries or missing:
            print()

    if args.weights:
        print(f"  —— 模型权重（{wt.MIRROR_SUBDIR}/）——")
        weight_entries = pack_weights(output, dry_run=args.dry_run)
        print()

    if args.dry_run:
        print("  （--dry-run：没有写任何文件）")
        return 0

    if not entries and not weight_entries:
        print("  什么都没打包。先看看哪些在本地：")
        print("    python scripts/fetch_datasets.py --status")
        print("    python scripts/fetch_datasets.py --weights")
        return 1

    manifest_entries = dict(entries)
    if args.skip_datasets:
        manifest_entries = _existing_datasets(output) or manifest_entries
    manifest = write_manifest(output, manifest_entries, weight_entries)
    total = sum(int(item["bytes"]) for item in entries.values())
    total += sum(int(item["bytes"]) for item in weight_entries.values())

    if args.json:
        print(json.dumps(
            {"manifest": str(manifest), "datasets": entries, "weights": weight_entries},
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print()
    _rule("=")
    print(f"  共 {len(entries)} 个数据包 + {len(weight_entries)} 个权重 / {_human(total)}")
    _rule("=")
    print(f"  清单  {manifest}")
    print()
    print("  下一步（三件事，做完别人就不需要 Kaggle 凭证了）：")
    print()
    print("  1) 把这个目录整个上传到任意静态托管：")
    print(f"       {output}")
    print("     GitHub Release、阿里云 OSS、腾讯 COS、内网文件服务器都行；")
    print("     要求只有一条：能按 <根地址>/<文件名> 直接 GET 到。")
    print()
    print("  2) 把根地址写成环境变量（写进用户变量才算永久）：")
    print("       set FISHERIES_DATASET_MIRROR=https://your-host/fish-datasets")
    print()
    print("  3) 验证：换一台没有 Kaggle 凭证的机器（或删掉 %USERPROFILE%\\.kaggle）跑")
    print("       python scripts/fetch_datasets.py --fetch all")
    print()
    print("  sha256 从 manifest.json 里读，注册表一行都不用改。")
    if weight_entries:
        print()
        print("  权重同理（不配 FISHERIES_WEIGHTS_MIRROR 时会自动用数据集镜像站）：")
        print("       python scripts/fetch_datasets.py --fetch-weights all")
        print("  评委核对 md5：")
        for name, item in sorted(weight_entries.items()):
            print(f'    {item["md5"]}  models/{Path(str(item["file"])).name}   # {name}')
        print("    也可以直接 md5sum -c models/checksums.md5")
    print()
    print("  想更严一点，可以把下面两行写进 src/fisheries/datasets.py 对应的 spec：")
    print()
    for name, item in sorted(entries.items()):
        print(f'    "{name}"：')
        print(f'        archive_url="https://your-host/fish-datasets/{item["file"]}",')
        print(f'        sha256="{item["sha256"]}",')
    print()

    if missing:
        print(f"  以下 {len(missing)} 个没有打包（本地没有或未注册）：")
        for name in missing:
            print(f"    · {name}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
