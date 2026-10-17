"""模型权重清单、校验、下载与降级提示的单元测试。

这里测的是三条契约：

1. **清单与校验文件必须一致**——``models/checksums.md5`` 是给评委看的副本，
   代码里的 ``WEIGHTS`` 是运行时用的；两者漂移就等于「文档骗人」。
2. **下载必须校验**——md5 或 sha256 不符就删掉，不许留下一个坏文件当权重用。
3. **降级必须出声**——找不到权重时打一行 WARNING，文案固定为
   ``MOCK_NOTICE``；同一个组件只打一次，否则每张图刷一遍等于没警告。

测试不依赖真实权重文件（它们不在仓库里），所以用 ``monkeypatch`` 把
``weights.PROJECT_ROOT`` 指到临时目录，并替换成一份自造的注册表。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pytest

from fisheries import weights as wt

#: 仓库根目录（测试跑在工程里，用来读真实的 .gitignore / checksums.md5）。
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 固定载荷，保证 md5 / sha256 断言可复现。
PAYLOAD = b"fake-weight-bytes" * 64


def _hashes(data: bytes) -> tuple[str, str]:
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    data: bytes = PAYLOAD,
    pinned: bool = True,
) -> wt.WeightSpec:
    """把注册表换成一个只含 ``fake`` 的临时注册表，根目录指到 ``tmp_path``。"""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    md5, sha256 = _hashes(data)
    monkeypatch.setattr(wt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(wt, "MODELS_DIR", models)
    monkeypatch.setattr(wt, "CHECKSUMS_FILE", models / "checksums.md5")
    item = wt.WeightSpec(
        name="fake",
        file="models/fake.pt",
        label="假权重",
        purpose="只用来测链路",
        fallback="退化到 mock",
        md5=md5,
        sha256=sha256,
        size=len(data),
        pinned=pinned,
    )
    monkeypatch.setattr(wt, "WEIGHTS", {"fake": item})
    monkeypatch.delenv(wt.MIRROR_BASE_ENV, raising=False)
    monkeypatch.delenv(wt.FALLBACK_MIRROR_ENV, raising=False)
    wt.reset_warnings()
    return item


def make_mirror(tmp_path: Path, data: bytes = PAYLOAD, name: str = "fake.pt") -> Path:
    """造一个「镜像站」目录，返回根地址。"""
    root = tmp_path / "mirror"
    target = root / wt.MIRROR_SUBDIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return root


# --------------------------------------------------------------------------- #
# 一、注册表
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_every_entry_has_a_distinct_file(self) -> None:
        files = [item.file for item in wt.WEIGHTS.values()]
        assert len(files) == len(set(files))
        assert all(path.startswith("models/") for path in files)

    def test_pinned_entries_carry_both_hashes(self) -> None:
        for item in wt.WEIGHTS.values():
            if not item.pinned:
                continue
            assert re.fullmatch(r"[0-9a-f]{32}", item.md5), item.name
            assert re.fullmatch(r"[0-9a-f]{64}", item.sha256), item.name
            assert item.size > 0, item.name

    def test_official_pretrained_entries_are_not_pinned(self) -> None:
        for name in ("yolov8n-cls", "yolov8n-seg"):
            item = wt.spec(name)
            assert item.pinned is False
            assert item.upstream.startswith("https://")
            assert item.hosted is False

    def test_unknown_name_says_what_is_known(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            wt.spec("yolov8-fish")
        assert "yolov8n-seg" in str(excinfo.value)

    def test_status_reports_missing_and_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        rows = wt.status()
        assert rows[0]["exists"] is False and rows[0]["bytes"] == 0
        item.path.write_bytes(PAYLOAD)
        rows = wt.status()
        assert rows[0]["exists"] is True
        assert rows[0]["size_ok"] is True


# --------------------------------------------------------------------------- #
# 二、校验
# --------------------------------------------------------------------------- #

class TestVerify:
    def test_missing_file_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        ok, detail = wt.verify("fake")
        assert ok is False
        assert "不存在" in detail

    def test_correct_file_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(PAYLOAD)
        ok, detail = wt.verify("fake")
        assert ok is True
        assert "md5" in detail and "sha256" in detail

    def test_wrong_size_is_caught_before_hashing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(PAYLOAD + b"extra")
        ok, detail = wt.verify("fake")
        assert ok is False
        assert "大小不符" in detail

    def test_same_size_wrong_bytes_is_caught_by_md5(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(b"x" * len(PAYLOAD))
        ok, detail = wt.verify("fake")
        assert ok is False
        assert "md5 不符" in detail

    def test_unpinned_entry_only_checks_size(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path, pinned=False)
        item.path.write_bytes(b"x" * len(PAYLOAD))  # 内容全错但大小对
        ok, detail = wt.verify("fake")
        assert ok is True
        assert "未钉哈希" in detail


# --------------------------------------------------------------------------- #
# 三、校验文件（进 git 的那份副本）
# --------------------------------------------------------------------------- #

class TestChecksums:
    def test_render_matches_the_committed_file(self) -> None:
        committed = REPO_ROOT / "models" / "checksums.md5"
        assert committed.is_file(), "models/checksums.md5 必须进 git"
        assert committed.read_text(encoding="utf-8") == wt.render_checksums()

    def test_every_pinned_hash_appears_in_the_rendered_text(self) -> None:
        text = wt.render_checksums()
        for item in wt.WEIGHTS.values():
            if item.pinned:
                assert f"{item.md5}  {item.file}" in text, item.name

    def test_committed_file_has_no_crlf(self) -> None:
        raw = (REPO_ROOT / "models" / "checksums.md5").read_bytes()
        assert b"\r\n" not in raw

    def test_read_checksums_parses_the_standard_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        target = tmp_path / "checksums.md5"
        target.write_text(
            "# 注释\n"
            "d41d8cd98f00b204e9800998ecf8427e  models/a.pt\n"
            "  0cc175b9c0f1b6a831c399e269772661   models/b.pt\n",
            encoding="utf-8",
        )
        table = wt.read_checksums(target)
        assert table == {
            "models/a.pt": "d41d8cd98f00b204e9800998ecf8427e",
            "models/b.pt": "0cc175b9c0f1b6a831c399e269772661",
        }

    def test_missing_file_reads_as_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        assert wt.read_checksums() == {}


# --------------------------------------------------------------------------- #
# 四、下载
# --------------------------------------------------------------------------- #

class TestDownload:
    def test_fetch_from_a_local_mirror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        root = make_mirror(tmp_path)
        monkeypatch.setenv(wt.MIRROR_BASE_ENV, str(root))
        got = wt.fetch("fake")
        assert got == item.path
        assert got.read_bytes() == PAYLOAD
        assert wt.verify("fake")[0] is True

    def test_second_call_uses_the_local_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        root = make_mirror(tmp_path)
        monkeypatch.setenv(wt.MIRROR_BASE_ENV, str(root))
        wt.fetch("fake")
        stamp = item.path.stat().st_mtime_ns
        (root / wt.MIRROR_SUBDIR / "fake.pt").unlink()  # 镜像站挂了也不影响
        assert wt.fetch("fake") == item.path
        assert item.path.stat().st_mtime_ns == stamp

    def test_corrupted_download_is_deleted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        root = make_mirror(tmp_path, data=b"y" * len(PAYLOAD))
        monkeypatch.setenv(wt.MIRROR_BASE_ENV, str(root))
        with pytest.raises(wt.WeightNotFoundError) as excinfo:
            wt.fetch("fake")
        assert "校验不过" in str(excinfo.value)
        assert not item.path.exists(), "坏文件不能留下"

    def test_missing_source_explains_the_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        with pytest.raises(wt.WeightNotFoundError) as excinfo:
            wt.fetch("fake")
        message = str(excinfo.value)
        assert wt.MIRROR_BASE_ENV in message
        assert "pack_dataset_mirror" in message

    def test_explicit_url_beats_the_mirror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        root = make_mirror(tmp_path, data=b"wrong" * len(PAYLOAD))
        monkeypatch.setenv(wt.MIRROR_BASE_ENV, str(root))
        alt = tmp_path / "elsewhere.pt"
        alt.write_bytes(PAYLOAD)
        assert wt.fetch("fake", url=str(alt)) == item.path
        assert wt.verify("fake")[0] is True

    def test_file_url_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        alt = tmp_path / "elsewhere.pt"
        alt.write_bytes(PAYLOAD)
        got = wt.fetch("fake", url=alt.as_uri())
        assert got.read_bytes() == PAYLOAD

    def test_mirror_falls_back_to_the_dataset_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        root = make_mirror(tmp_path)
        monkeypatch.setenv(wt.FALLBACK_MIRROR_ENV, str(root))
        assert wt.mirror_base() == str(root).rstrip("/")
        url = wt.url_for("fake")
        # 本机路径的斜杠方向随平台，这里只校验「根 + weights/ + 文件名」这三段
        assert url.startswith(str(root).rstrip("/"))
        assert url.endswith(f"{wt.MIRROR_SUBDIR}/fake.pt")

    def test_fetch_all_collects_failures_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        results = wt.fetch_all()
        assert "失败" in results["fake"]

    def test_download_leaves_no_part_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        make_mirror(tmp_path)
        monkeypatch.setenv(wt.MIRROR_BASE_ENV, str(tmp_path / "mirror"))
        wt.fetch("fake")
        assert list(item.path.parent.glob("*.part")) == []


# --------------------------------------------------------------------------- #
# 五、降级提示
# --------------------------------------------------------------------------- #

class TestMockNotice:
    def test_bare_notice_is_the_agreed_sentence(self) -> None:
        assert wt.mock_notice() == wt.MOCK_NOTICE
        assert wt.MOCK_NOTICE == "权重缺失，将启用 mock 模拟模式"

    def test_named_notice_says_file_and_how_to_get_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        text = wt.mock_notice("fake")
        assert "models/fake.pt" in text
        assert "--fetch-weights fake" in text
        assert wt.MOCK_NOTICE in text

    def test_warn_mock_logs_only_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        with caplog.at_level(logging.WARNING, logger="fisheries.weights"):
            wt.warn_mock("component", "fake")
            wt.warn_mock("component", "fake")
        hits = [r for r in caplog.records if "component" in r.getMessage()]
        assert len(hits) == 1
        assert wt.MOCK_NOTICE in hits[0].getMessage()

    def test_reset_warnings_lets_it_speak_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        with caplog.at_level(logging.WARNING, logger="fisheries.weights"):
            wt.warn_mock("component", "fake")
            wt.reset_warnings()
            wt.warn_mock("component", "fake")
        assert len([r for r in caplog.records if "component" in r.getMessage()]) == 2


# --------------------------------------------------------------------------- #
# 六、接线：工具层与接口层必须喊出来
# --------------------------------------------------------------------------- #

class TestWiring:
    def test_missing_weights_make_the_tool_say_mock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from fisheries.tools import fish_disease

        install_fake(monkeypatch, tmp_path)
        monkeypatch.setattr(fish_disease, "_MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setattr(fish_disease, "_WEIGHTS_CANDIDATES", ("nope.pt",))
        with caplog.at_level(logging.WARNING, logger="fisheries.weights"):
            backend, weights_path = fish_disease._resolve_backend(None)
        assert (backend, weights_path) == ("mock", None)
        assert wt.MOCK_NOTICE in caplog.text

    def test_missing_lesion_weights_speaks_without_calling_it_mock(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from fisheries.tools import fish_disease

        install_fake(monkeypatch, tmp_path)
        monkeypatch.setattr(fish_disease, "_MODELS_DIR", str(tmp_path / "models"))
        monkeypatch.setattr(fish_disease, "LESION_WEIGHTS_CANDIDATES", ("nope-seg.pt",))
        with caplog.at_level(logging.WARNING, logger="fisheries.weights"):
            assert fish_disease._resolve_lesion_weights(None) is None
        assert "病灶" in caplog.text
        assert wt.MOCK_NOTICE not in caplog.text

    def test_find_returns_none_for_missing_local_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        assert wt.find("fake") is None
        assert wt.missing() == ["fake"]


# --------------------------------------------------------------------------- #
# 七、仓库配置与命令行
# --------------------------------------------------------------------------- #

class TestRepoConfig:
    def test_gitignore_re_includes_the_checksum_file(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "models/*" in text
        assert "!models/checksums.md5" in text
        # 整目录忽略会让 ! 失效，所以不允许再出现裸的 models/ 行
        assert not re.search(r"^models/\s*$", text, flags=re.MULTILINE)

    def test_fetch_datasets_exposes_the_weight_commands(self) -> None:
        text = (REPO_ROOT / "scripts" / "fetch_datasets.py").read_text(encoding="utf-8")
        for flag in ("--weights", "--fetch-weights", "--write-checksums"):
            assert flag in text


class TestCommandLine:
    def _module(self):
        import importlib.util

        path = REPO_ROOT / "scripts" / "fetch_datasets.py"
        spec = importlib.util.spec_from_file_location("_fetch_datasets_under_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _packer(self):
        import importlib.util

        path = REPO_ROOT / "scripts" / "pack_dataset_mirror.py"
        spec = importlib.util.spec_from_file_location("_packer_under_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_weights_json_is_machine_readable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        module = self._module()
        assert module.main(["--weights", "--json"]) in (0, 1)
        payload = json.loads(capsys.readouterr().out)
        assert payload["items"][0]["name"] == "fake"
        assert "mirror" in payload

    def test_skip_datasets_requires_weights_flag(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module = self._packer()
        assert module.main(["--skip-datasets"]) == 2
        assert "--weights" in capsys.readouterr().out

    def test_packer_dry_run_lists_weights_without_writing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(PAYLOAD)
        module = self._packer()
        out = tmp_path / "dist"
        assert module.main(["--skip-datasets", "--weights", "--dry-run", "--out", str(out)]) == 0
        text = capsys.readouterr().out
        assert "fake" in text and "没有写任何文件" in text
        assert not (out / wt.MIRROR_SUBDIR).exists()

    def test_packer_copies_weights_and_records_md5(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(PAYLOAD)
        module = self._packer()
        out = tmp_path / "dist"
        assert module.main(["--skip-datasets", "--weights", "--out", str(out)]) == 0
        copied = out / wt.MIRROR_SUBDIR / "fake.pt"
        assert copied.read_bytes() == PAYLOAD
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["weights"]["fake"]["md5"] == item.md5
        assert manifest["weights"]["fake"]["file"] == f"{wt.MIRROR_SUBDIR}/fake.pt"
        assert manifest["datasets"] == {}

    def test_write_checksums_dry_run_does_not_touch_the_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        install_fake(monkeypatch, tmp_path)
        module = self._module()
        assert module.main(["--write-checksums", "--dry-run"]) == 0
        assert wt.CHECKSUMS_FILE.exists() is False
        assert "fake" in capsys.readouterr().out

    def test_write_checksums_writes_the_rendered_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(PAYLOAD)
        module = self._module()
        assert module.main(["--write-checksums"]) == 0
        assert wt.CHECKSUMS_FILE.read_text(encoding="utf-8") == wt.render_checksums()
        assert "已写入" in capsys.readouterr().out

    def test_write_checksums_refuses_when_the_file_disagrees(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """重训后注册表没更新时，宁可不写，也不能写出一张对不上的校验单。"""
        item = install_fake(monkeypatch, tmp_path)
        item.path.write_bytes(b"trained-again")
        module = self._module()
        assert module.main(["--write-checksums"]) == 1
        text = capsys.readouterr().out
        assert "weights.py" in text
        assert "应更新为 md5=" in text
        assert not wt.CHECKSUMS_FILE.exists()
