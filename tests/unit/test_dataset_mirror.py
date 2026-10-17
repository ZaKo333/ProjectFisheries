"""数据集镜像：打包、下载回退、sha256 校验。

被测的是「没有 Kaggle 凭证怎么拿到数据」这条链：

    打包脚本  →  dist/mirror/<名字>.zip + manifest.json
    上传      →  任意静态托管（本测用临时目录模拟）
    fetch     →  先试 kaggle，失败自动切镜像，不需要 kagglehub

三件事必须成立，否则镜像这条路等于没做：

1. **可复现**：同一份数据重复打包得到同一个 sha256——不然校验值没法写进注册表。
2. **不依赖 kagglehub**：测试环境本来就没装它，走的正是「kaggle 失败」这条路。
3. **错了就拒绝**：sha256 对不上、zip 里有目录穿越，都必须是明确的错误而不是猜。
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

from fisheries import datasets as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACK_PATH = PROJECT_ROOT / "scripts" / "pack_dataset_mirror.py"
SETUP_PATH = PROJECT_ROOT / "scripts" / "setup_env.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pack = _load(PACK_PATH, "pack_dataset_mirror_under_test")


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #

DATASET_NAME = "mirror_demo"
DATASET_ENTRY = "mirror_demo"


def _write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf-8")
    path.write_bytes(content)


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    """一份假的本地数据集：``<root>/mirror_demo/Test/a.jpg``。"""
    root = tmp_path / "local"
    _write(root / DATASET_ENTRY / "Test" / "a.jpg", b"fake-jpeg-bytes")
    _write(root / DATASET_ENTRY / "Test" / "b.jpg", b"another-fake-jpeg")
    return root / DATASET_ENTRY


@pytest.fixture()
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把 datasets 模块的写入位置指到临时目录，并注册一个假数据集。"""
    local_root = tmp_path / "installed"
    index_path = tmp_path / "dataset_index.json"
    monkeypatch.setattr(ds, "LOCAL_ROOT", local_root)
    monkeypatch.setattr(ds, "INDEX_PATH", index_path)
    spec = ds.DatasetSpec(
        name=DATASET_NAME,
        entry=DATASET_ENTRY,
        kaggle_id="someone/mirror-demo",
        label="镜像测试用",
        required=("Test",),
    )
    monkeypatch.setitem(ds.DATASETS, DATASET_NAME, spec)
    return spec


@pytest.fixture()
def mirror(tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """打好包并把它当镜像站用（``FISHERIES_DATASET_MIRROR`` 指过去）。"""
    out = tmp_path / "mirror"
    entry = pack.pack_one(source, DATASET_ENTRY, out / f"{DATASET_NAME}.zip", level=6)
    pack.write_manifest(out, {DATASET_NAME: entry})
    monkeypatch.setenv(ds.MIRROR_BASE_ENV, str(out))
    return out


# --------------------------------------------------------------------------- #
# 打包
# --------------------------------------------------------------------------- #

class TestPack:
    def test_repack_is_byte_identical(self, tmp_path: Path, source: Path) -> None:
        """同一份数据打两次必须是同一个 sha256——否则校验值每次都不一样。"""
        first = pack.pack_one(source, DATASET_ENTRY, tmp_path / "a.zip", level=6)
        second = pack.pack_one(source, DATASET_ENTRY, tmp_path / "b.zip", level=6)
        assert first["sha256"] == second["sha256"]
        assert first["bytes"] == second["bytes"]
        assert first["files"] == 2

    def test_members_are_relative_and_complete(
        self, tmp_path: Path, source: Path
    ) -> None:
        pack.pack_one(source, DATASET_ENTRY, tmp_path / "a.zip", level=6)
        with zipfile.ZipFile(tmp_path / "a.zip") as bundle:
            assert bundle.namelist() == ["Test/a.jpg", "Test/b.jpg"]

    def test_manifest_records_sha256_and_count(
        self, tmp_path: Path, source: Path
    ) -> None:
        entry = pack.pack_one(
            source, DATASET_ENTRY, tmp_path / f"{DATASET_NAME}.zip", level=6
        )
        path = pack.write_manifest(tmp_path, {DATASET_NAME: entry})
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        item = payload["datasets"][DATASET_NAME]
        assert payload["schema"] == 1
        assert item["file"] == f"{DATASET_NAME}.zip"
        assert len(item["sha256"]) == 64
        assert item["files"] == 2

    def test_single_file_dataset_uses_entry_name(
        self, tmp_path: Path, source: Path
    ) -> None:
        """单文件数据集（entry 是文件名）在压缩包根目录下就叫那个名字。"""
        csv = tmp_path / "one.csv"
        _write(csv, "a,b\n1,2\n")
        entry = pack.pack_one(csv, "one.csv", tmp_path / "one.zip", level=6)
        with zipfile.ZipFile(tmp_path / "one.zip") as bundle:
            assert bundle.namelist() == ["one.csv"]
        assert entry["files"] == 1


# --------------------------------------------------------------------------- #
# 下载：kaggle 失败 → 镜像
# --------------------------------------------------------------------------- #

class TestFallback:
    def test_kaggle_missing_falls_back_to_mirror(
        self, installed: ds.DatasetSpec, mirror: Path
    ) -> None:
        """当前环境没装 kagglehub，所以这条断言走的就是「kaggle 失败」分支。"""
        assert importlib.util.find_spec("kagglehub") is None
        path = ds.fetch(DATASET_NAME)
        assert path == ds.LOCAL_ROOT / DATASET_ENTRY
        assert (path / "Test" / "a.jpg").read_bytes() == b"fake-jpeg-bytes"

    def test_second_call_returns_cached_path(
        self, installed: ds.DatasetSpec, mirror: Path
    ) -> None:
        first = ds.fetch(DATASET_NAME)
        second = ds.fetch(DATASET_NAME)
        assert first == second

    def test_mirror_first_order_also_works(
        self, installed: ds.DatasetSpec, mirror: Path
    ) -> None:
        path = ds.fetch(DATASET_NAME, mirror_first=True)
        assert (path / "Test" / "b.jpg").is_file()

    def test_failure_message_lists_both_paths_and_next_steps(
        self, installed: ds.DatasetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ds.MIRROR_BASE_ENV, raising=False)
        with pytest.raises(ds.DatasetNotFoundError) as excinfo:
            ds.fetch(DATASET_NAME)
        text = str(excinfo.value)
        # 「两条路都不通」必须真的两条都列出来，哪怕镜像那一路只是「没配」。
        assert "kaggle：" in text
        assert "镜像：" in text
        assert "没配" in text
        assert ds.MIRROR_BASE_ENV in text
        assert "pack_dataset_mirror.py" in text

    def test_sha256_mismatch_is_rejected(
        self, installed: ds.DatasetSpec, mirror: Path
    ) -> None:
        archive = mirror / f"{DATASET_NAME}.zip"
        archive.write_bytes(archive.read_bytes() + b"tampered")
        with pytest.raises(ds.DatasetNotFoundError) as excinfo:
            ds.fetch(DATASET_NAME)
        assert "sha256" in str(excinfo.value)
        assert not (ds.LOCAL_ROOT / DATASET_ENTRY).exists()

    def test_registry_sha256_wins_over_manifest(
        self,
        tmp_path: Path,
        installed: ds.DatasetSpec,
        mirror: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """注册表里写了 sha256 就以它为准（清单被换掉也拦得住）。"""
        wrong = ds.DatasetSpec(
            name=DATASET_NAME,
            entry=DATASET_ENTRY,
            kaggle_id="someone/mirror-demo",
            required=("Test",),
            sha256="0" * 64,
        )
        monkeypatch.setitem(ds.DATASETS, DATASET_NAME, wrong)
        with pytest.raises(ds.DatasetNotFoundError) as excinfo:
            ds.fetch(DATASET_NAME)
        # 报错正文被 _one_line 截断过，所以只断言结论性措辞；完整的期望值
        # 在 sha256 校验那一步的原始消息里。
        assert "sha256 对不上" in str(excinfo.value)


class TestLocalSources:
    def test_download_accepts_plain_path(self, tmp_path: Path, source: Path) -> None:
        archive = tmp_path / "x.zip"
        pack.pack_one(source, DATASET_ENTRY, archive, level=6)
        dest = tmp_path / "copy.zip"
        used = ds._download(str(archive), dest)
        assert used == str(archive)
        assert dest.read_bytes() == archive.read_bytes()

    def test_download_accepts_file_url(self, tmp_path: Path, source: Path) -> None:
        archive = tmp_path / "y.zip"
        pack.pack_one(source, DATASET_ENTRY, archive, level=6)
        dest = tmp_path / "copy2.zip"
        used = ds._download(archive.as_uri(), dest)
        assert used == archive.as_uri()
        assert dest.is_file()

    def test_missing_local_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ds.DatasetNotFoundError):
            ds._download(str(tmp_path / "nope.zip"), tmp_path / "out.zip")


class TestZipSafety:
    def test_traversal_members_are_skipped(
        self, tmp_path: Path, installed: ds.DatasetSpec
    ) -> None:
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("Test/ok.jpg", b"ok")
            bundle.writestr("../evil.txt", b"evil")
            bundle.writestr("/abs.txt", b"abs")
            bundle.writestr("C:/drive.txt", b"drive")
        staging = tmp_path / "staging"
        written = ds._extract_archive(archive, staging, installed)
        assert written == 1
        assert (staging / "Test" / "ok.jpg").is_file()
        assert not (tmp_path / "evil.txt").exists()
        # Windows 上 Path("/abs.txt").is_absolute() 是 False，光靠 is_absolute 拦不住。
        assert not Path("/abs.txt").exists()
        assert not Path("C:/drive.txt").exists()

    def test_single_file_entry_lands_at_entry_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``entry`` 是文件名时，解出来的文件要搬到 entry，而不是变成目录。"""
        monkeypatch.setattr(ds, "LOCAL_ROOT", tmp_path / "installed")
        monkeypatch.setattr(ds, "INDEX_PATH", tmp_path / "index.json")
        spec = ds.DatasetSpec(name="flat_demo", entry="flat.csv", label="单文件")
        monkeypatch.setitem(ds.DATASETS, "flat_demo", spec)

        mirror = tmp_path / "mirror"
        _write(tmp_path / "src" / "flat.csv", "x,y\n1,2\n")
        entry = pack.pack_one(tmp_path / "src" / "flat.csv", "flat.csv",
                              mirror / "flat_demo.zip", level=6)
        pack.write_manifest(mirror, {"flat_demo": entry})
        monkeypatch.setenv(ds.MIRROR_BASE_ENV, str(mirror))

        path = ds.fetch("flat_demo")
        assert path == tmp_path / "installed" / "flat.csv"
        assert path.is_file()
        assert "x,y" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 一键配置里的这一步
# --------------------------------------------------------------------------- #

class TestSetupStep:
    def test_script_declares_dataset_step(self) -> None:
        text = SETUP_PATH.read_text(encoding="utf-8")
        assert "def step_datasets(" in text
        assert "step_datasets(args)" in text
        assert "--skip-datasets" in text
        assert "--dataset-mirror" in text

    def test_step_is_skippable(self) -> None:
        setup_env = _load(SETUP_PATH, "setup_env_mirror_test")
        args = setup_env.build_parser().parse_args(["--skip-datasets"])
        assert args.skip_datasets is True
        assert setup_env.step_datasets(args) is True

    def test_mirror_flag_is_forwarded_as_env(self) -> None:
        text = SETUP_PATH.read_text(encoding="utf-8")
        assert '"FISHERIES_DATASET_MIRROR": args.dataset_mirror' in text

    def test_bat_core_paths_are_ascii(self) -> None:
        """setup.bat 仍必须是纯 ASCII（cmd.exe 的硬约束）。"""
        data = (PROJECT_ROOT / "setup.bat").read_bytes()
        data.decode("ascii")
