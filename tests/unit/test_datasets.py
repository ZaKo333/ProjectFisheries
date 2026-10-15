"""数据集注册表与路径解析（``fisheries.datasets``）的单元测试。

这一层是「工程不带数据集」这件事的地基，所以测试重点不在「能找到文件」，
而在**每条查找路径都按声明的优先级生效**，以及**找不到时给出的是可操作的
错误**而不是一句「找不到」。

所有涉及磁盘的用例都用 ``tmp_path`` + ``monkeypatch`` 把常量改掉，因此：

* 有无本机数据集，测试结果一致（不依赖 ``D:\\设置\\项目\\Agent\\数据集``）；
* 不会真的联网下载（``fetch`` 只在「已能找到」或「kagglehub 缺失」两条
  分支上被测，前者不触发下载，后者不触发下载）。
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from fisheries import datasets as ds

# --------------------------------------------------------------------------- #
# 夹具：把三个根目录和索引文件全部搬到 tmp_path，环境变量清干净
# --------------------------------------------------------------------------- #


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """一个「什么都没有」的数据集环境。

    返回一个可写的 ``root``，往里面摆目录就等于造数据。
    """
    root = tmp_path / "data-root"
    root.mkdir()
    local = tmp_path / "project" / "data" / "datasets"
    legacy = tmp_path / "legacy" / "数据集"
    cache = tmp_path / "kaggle"

    monkeypatch.delenv(ds.DATA_ROOT_ENV, raising=False)
    monkeypatch.setattr(ds, "LOCAL_ROOT", local)
    monkeypatch.setattr(ds, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(ds, "INDEX_PATH", tmp_path / "data" / "dataset_index.json")
    monkeypatch.setattr(ds, "KAGGLE_CACHE_ROOT", cache)
    return root


def _make_dataset(root: Path, name: str, *parts: str) -> Path:
    """在 ``root`` 下按注册表的 entry 造出这个数据集（含 required 子路径）。"""
    spec = ds.DATASETS[name]
    base = root / spec.entry
    base.mkdir(parents=True, exist_ok=True)
    for part in (*spec.required, *parts):
        target = base / part
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
    return base


def _make_kaggle_cache(cache: Path, name: str, version: str = "3") -> Path:
    spec = ds.DATASETS[name]
    owner, slug = spec.kaggle_id.split("/", 1)
    base = cache / owner / slug / "versions" / version / spec.entry
    base.mkdir(parents=True, exist_ok=True)
    for part in spec.required:
        (base / part).mkdir(parents=True, exist_ok=True)
    return base


# --------------------------------------------------------------------------- #
# 注册表自身的完整性——注册表写错，所有调用方一起错
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_keys_match_spec_names(self):
        for key, spec in ds.DATASETS.items():
            assert spec.name == key, f"{key} 的 name 字段是 {spec.name}"

    def test_entries_are_relative(self):
        for spec in ds.DATASETS.values():
            assert spec.entry, f"{spec.name} 没有 entry"
            assert not Path(spec.entry).is_absolute(), f"{spec.name} 的 entry 是绝对路径"

    def test_every_dataset_has_a_source(self):
        # 没 kaggle id 就必须有 url，否则用户拿到「找不到」也不知道去哪下。
        for spec in ds.DATASETS.values():
            assert spec.kaggle_id or spec.url, f"{spec.name} 没有任何来源信息"

    def test_kaggle_ids_are_owner_slug(self):
        for spec in ds.DATASETS.values():
            if not spec.kaggle_id:
                continue
            parts = spec.kaggle_id.split("/")
            assert len(parts) == 2, f"{spec.name} 的 kaggle_id 不是 owner/slug：{spec.kaggle_id}"
            assert all(parts), f"{spec.name} 的 kaggle_id 有空段：{spec.kaggle_id}"

    def test_code_dependencies_are_declared(self):
        for name in ("fish_disease", "aquaponics_pond"):
            assert ds.DATASETS[name].used_by, f"{name} 被代码使用但没写 used_by"

    def test_retired_datasets_are_not_declared_as_used(self):
        # china_water_pollution 与 river_do 是 M8 前两轮用的数据，现已停用：
        # 条目保留备查，但代码里已经没有读取它们的路径，所以 used_by 必须为空。
        for name in ("china_water_pollution", "river_do"):
            assert ds.DATASETS[name].used_by == (), f"{name} 已停用但还标着 used_by"

    def test_optional_datasets_are_not_declared_as_used(self):
        # 备选数据集如果被标成 used_by，说明注册表在骗人。
        for name in ("water_potability", "fish_market", "soil_ammonia_15n"):
            assert ds.DATASETS[name].used_by == ()

    def test_list_datasets_matches_registry(self):
        assert [s.name for s in ds.list_datasets()] == list(ds.DATASETS)

    def test_title_falls_back_to_name(self):
        spec = ds.DatasetSpec(name="x", entry="x")
        assert spec.title == "x"
        assert ds.DatasetSpec(name="x", entry="x", label="L").title == "L"

    def test_download_command_uses_fetch_script(self):
        cmd = ds.DATASETS["fish_disease"].download_command()
        assert "fetch_datasets.py" in cmd
        assert "--fetch fish_disease" in cmd

    def test_download_command_without_kaggle_id_points_at_the_url(self):
        spec = ds.DatasetSpec(name="x", entry="x", url="https://example.com/d")
        assert "https://example.com/d" in spec.download_command()

    def test_kaggle_snippet_compiles(self):
        snippet = ds.DATASETS["fish_disease"].kaggle_snippet()
        compile(snippet, "<snippet>", "exec")
        assert "dataset_download" in snippet

    def test_kaggle_snippet_empty_without_a_handle(self):
        assert ds.DatasetSpec(name="x", entry="x").kaggle_snippet() == ""


# --------------------------------------------------------------------------- #
# 名字不合法
# --------------------------------------------------------------------------- #


class TestUnknownName:
    def test_unknown_name_lists_the_registered_ones(self):
        with pytest.raises(KeyError) as exc:
            ds.find("no_such_dataset")
        message = str(exc.value)
        assert "no_such_dataset" in message
        assert "fish_disease" in message

    def test_require_and_resolve_also_reject_unknown_names(self):
        with pytest.raises(KeyError):
            ds.require("nope")
        with pytest.raises(KeyError):
            ds.resolve("nope", "a.csv")


# --------------------------------------------------------------------------- #
# 找不到时的错误——必须可操作
# --------------------------------------------------------------------------- #


class TestNotFound:
    def test_error_is_a_file_not_found_error(self):
        # 兼容性契约：原来 except FileNotFoundError 的调用方不用改。
        assert issubclass(ds.DatasetNotFoundError, FileNotFoundError)

    def test_error_carries_the_dataset_name(self):
        err = ds.DatasetNotFoundError("river_do", "boom")
        assert err.name == "river_do"

    def test_find_returns_none_require_raises(self, isolated):
        assert ds.find("river_do") is None
        with pytest.raises(ds.DatasetNotFoundError):
            ds.require("river_do")

    def test_message_contains_every_searched_location(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.require("river_do")
        message = str(exc.value)
        for label, _root in ds._roots():  # noqa: SLF001 - 断言的就是这份清单
            assert label in message
        # 给出实际查找路径，而不是一句「找不到」。
        assert str(ds.LEGACY_ROOT / ds.DATASETS["river_do"].entry) in message
        assert str(ds.LOCAL_ROOT / ds.DATASETS["river_do"].entry) in message

    def test_message_contains_the_download_command(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.require("river_do")
        assert "--fetch river_do" in str(exc.value)

    def test_message_explains_the_env_var_escape_hatch(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.require("river_do")
        assert ds.DATA_ROOT_ENV in str(exc.value)

    def test_message_points_at_the_status_command(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.require("river_do")
        assert "--status" in str(exc.value)

    def test_message_includes_the_reference_url(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.require("freshwater_fish_disease_asia")
        assert "platform.ultralytics.com" in str(exc.value)


# --------------------------------------------------------------------------- #
# 五条解析路径，逐条验证优先级
# --------------------------------------------------------------------------- #


class TestResolutionOrder:
    def test_env_var_is_searched_first(self, isolated, monkeypatch):
        _make_dataset(isolated, "river_do")
        _make_dataset(ds.LEGACY_ROOT, "river_do")
        monkeypatch.setenv(ds.DATA_ROOT_ENV, str(isolated))
        status = ds.status("river_do")[0]
        assert status.available
        assert status.source == f"env:{ds.DATA_ROOT_ENV}"
        assert status.path == isolated / ds.DATASETS["river_do"].entry

    def test_local_root_beats_legacy(self, isolated, monkeypatch):
        _make_dataset(ds.LOCAL_ROOT, "river_do")
        _make_dataset(ds.LEGACY_ROOT, "river_do")
        monkeypatch.setenv(ds.DATA_ROOT_ENV, str(isolated))
        status = ds.status("river_do")[0]
        assert status.source == "工程 data/datasets"

    def test_legacy_root_is_still_supported(self, isolated):
        # 向后兼容：改造前数据就摆在 <工程上级>/数据集，必须照样找得到。
        _make_dataset(ds.LEGACY_ROOT, "river_do")
        status = ds.status("river_do")[0]
        assert status.available
        assert status.source == "历史目录 数据集/"

    def test_index_file_is_used_when_roots_miss(self, isolated):
        elsewhere = _make_dataset(isolated, "river_do")
        ds._record("river_do", elsewhere)  # noqa: SLF001 - 模拟 fetch 之后的状态
        status = ds.status("river_do")[0]
        assert status.available
        assert status.source == "索引 dataset_index.json"

    def test_kaggle_cache_is_the_last_resort(self, isolated):
        _make_kaggle_cache(ds.KAGGLE_CACHE_ROOT, "river_do")
        status = ds.status("river_do")[0]
        assert status.available
        assert status.source == "kagglehub 缓存"

    def test_roots_without_env_var_has_two_entries(self, isolated):
        labels = [label for label, _ in ds._roots()]  # noqa: SLF001
        assert labels == ["工程 data/datasets", "历史目录 数据集/"]


class TestRequiredPaths:
    def test_entry_alone_is_not_enough(self, isolated):
        # fish_disease 的 required 是 ("Test",)：只有空目录不算可用。
        (ds.LEGACY_ROOT / "Fish Disease Dataset").mkdir(parents=True)
        assert ds.find("fish_disease") is None

    def test_required_subpath_makes_it_usable(self, isolated):
        _make_dataset(ds.LEGACY_ROOT, "fish_disease")
        assert ds.find("fish_disease") is not None

    def test_required_can_name_a_file(self, isolated):
        # river_do 的 required 是 ("train.csv",)。
        base = ds.LEGACY_ROOT / ds.DATASETS["river_do"].entry
        base.mkdir(parents=True)
        assert ds.find("river_do") is None
        (base / "train.csv").write_text("a,b\n", encoding="utf-8")
        assert ds.find("river_do") is not None

    def test_entry_may_be_a_single_file(self, isolated):
        # china_water_pollution 的 entry 就是一个散落在根目录的 CSV。
        target = ds.LEGACY_ROOT / ds.DATASETS["china_water_pollution"].entry
        target.parent.mkdir(parents=True)
        target.write_text("a\n", encoding="utf-8")
        assert ds.find("china_water_pollution") == target


class TestResolve:
    def test_resolve_appends_parts(self, isolated):
        base = _make_dataset(ds.LEGACY_ROOT, "fish_disease", "Test/A")
        assert ds.resolve("fish_disease", "Test", "A") == base / "Test" / "A"

    def test_resolve_without_parts_returns_the_base(self, isolated):
        base = _make_dataset(ds.LEGACY_ROOT, "river_do")
        assert ds.resolve("river_do") == base


# --------------------------------------------------------------------------- #
# default_path：模块级默认值专用，import 期绝不许失败
# --------------------------------------------------------------------------- #


class TestDefaultPath:
    def test_never_raises_even_when_missing(self, isolated):
        path = ds.default_path("river_do")
        assert isinstance(path, Path)

    def test_returns_the_real_path_when_present(self, isolated):
        base = _make_dataset(ds.LEGACY_ROOT, "river_do")
        assert ds.default_path("river_do") == base

    def test_placeholder_prefers_the_legacy_location(self, isolated):
        # 占位路径挑「用户最可能把数据放这儿」的地方：历史目录存在就用历史目录。
        # 这里刻意只建 entry 目录、不建 required 的子路径，所以 find() 仍然找不到，
        # 但「该放哪儿」的提示应该指向历史目录。
        (ds.LEGACY_ROOT / ds.DATASETS["river_do"].entry).mkdir(parents=True)
        assert ds.find("river_do") is None
        assert ds.default_path("river_do") == ds.LEGACY_ROOT / ds.DATASETS["river_do"].entry

    def test_placeholder_falls_back_to_local_root(self, isolated):
        # 哪儿都没有时给工程内的导入目录——那是 fetch 的默认落点。
        assert ds.default_path("river_do") == ds.LOCAL_ROOT / ds.DATASETS["river_do"].entry

    def test_placeholder_survives_a_missing_legacy_root(self, isolated, monkeypatch):
        monkeypatch.setattr(ds, "LEGACY_ROOT", isolated / "nope" / "deeper")
        assert ds.default_path("river_do") == ds.LOCAL_ROOT / ds.DATASETS["river_do"].entry


# --------------------------------------------------------------------------- #
# 索引文件
# --------------------------------------------------------------------------- #


class TestIndex:
    def test_roundtrip(self, isolated):
        ds._record("river_do", Path("C:/somewhere"))  # noqa: SLF001
        assert ds._read_index()["river_do"] == str(Path("C:/somewhere"))  # noqa: SLF001

    def test_record_preserves_other_entries(self, isolated):
        ds._record("river_do", Path("a"))  # noqa: SLF001
        ds._record("fish_disease", Path("b"))  # noqa: SLF001
        index = ds._read_index()  # noqa: SLF001
        assert index == {"river_do": "a", "fish_disease": "b"}

    def test_broken_json_is_ignored_not_fatal(self, isolated):
        ds.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        ds.INDEX_PATH.write_text("{not json", encoding="utf-8")
        assert ds._read_index() == {}  # noqa: SLF001

    def test_non_dict_json_is_ignored(self, isolated):
        ds.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        ds.INDEX_PATH.write_text("[1, 2]", encoding="utf-8")
        assert ds._read_index() == {}  # noqa: SLF001

    def test_a_stale_index_path_is_not_trusted(self, isolated):
        # 索引里记的路径已经不存在了（比如盘拔了），必须继续往下找，而不是崩。
        ds._record("river_do", isolated / "gone")  # noqa: SLF001
        assert ds.find("river_do") is None

    def test_index_path_lives_under_the_project(self):
        assert ds.index_path().name == "dataset_index.json"
        assert ds.index_path().parent.name == "data"


# --------------------------------------------------------------------------- #
# kagglehub 缓存扫描（只读、不联网）
# --------------------------------------------------------------------------- #


class TestKaggleCache:
    def test_missing_cache_dir_gives_empty(self, isolated):
        assert ds._kaggle_cache_candidates("owner/slug") == []  # noqa: SLF001

    def test_handle_without_slash_gives_empty(self, isolated):
        assert ds._kaggle_cache_candidates("noslug") == []  # noqa: SLF001
        assert ds._kaggle_cache_candidates("") == []  # noqa: SLF001

    def test_versions_are_sorted_numerically_desc(self, isolated):
        versions = ds.KAGGLE_CACHE_ROOT / "owner" / "slug" / "versions"
        for name in ("1", "2", "10", "beta"):
            (versions / name).mkdir(parents=True)
        found = [p.name for p in ds._kaggle_cache_candidates("owner/slug")]  # noqa: SLF001
        assert found == ["10", "2", "1", "beta"]

    def test_owner_slug_is_split_on_the_first_slash(self, isolated):
        versions = ds.KAGGLE_CACHE_ROOT / "owner" / "slug" / "versions" / "1"
        versions.mkdir(parents=True)
        assert ds._kaggle_cache_candidates("owner/slug") == [versions]  # noqa: SLF001


# --------------------------------------------------------------------------- #
# fetch：能不下就不下
# --------------------------------------------------------------------------- #


class TestFetch:
    def test_existing_local_copy_short_circuits_the_download(self, isolated):
        base = _make_dataset(ds.LEGACY_ROOT, "river_do")
        # kagglehub 没装也应该成功——因为根本不需要它。
        assert ds.fetch("river_do") == base

    KAGGLEHUB_INSTALLED = importlib.util.find_spec("kagglehub") is not None

    @pytest.mark.skipif(
        KAGGLEHUB_INSTALLED, reason="已装 kagglehub 时 fetch 会真的联网下载"
    )
    def test_missing_kagglehub_gives_an_install_hint(self, isolated):
        with pytest.raises(ds.DatasetNotFoundError) as exc:
            ds.fetch("river_do")
        message = str(exc.value)
        assert "kagglehub" in message
        assert "pip install" in message
        # 也要告诉人「不装也能用」的出路。
        assert ds.DATA_ROOT_ENV in message

    def test_fetch_on_unknown_name_still_raises_key_error(self, isolated):
        with pytest.raises(KeyError):
            ds.fetch("nope")


# --------------------------------------------------------------------------- #
# 状态与体积
# --------------------------------------------------------------------------- #


class TestStatus:
    def test_one_row_per_dataset(self, isolated):
        rows = ds.status_all()
        assert [r.spec.name for r in rows] == list(ds.DATASETS)

    def test_single_name(self, isolated):
        rows = ds.status("river_do")
        assert len(rows) == 1
        assert rows[0].spec.name == "river_do"

    def test_unavailable_row(self, isolated):
        row = ds.status("river_do")[0]
        assert not row.available
        assert row.source == ""
        assert row.size_text == "—"

    def test_size_is_only_measured_on_request(self, isolated):
        _make_dataset(ds.LEGACY_ROOT, "river_do")
        assert ds.status("river_do")[0].size_bytes is None

    def test_size_of_a_file_entry(self, isolated):
        target = ds.LEGACY_ROOT / ds.DATASETS["china_water_pollution"].entry
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * 2048)
        row = ds.status("china_water_pollution", with_size=True)[0]
        assert row.size_bytes == 2048

    def test_size_of_a_directory_entry(self, isolated):
        base = _make_dataset(ds.LEGACY_ROOT, "river_do")
        (base / "train.csv").write_bytes(b"x" * 1000)
        (base / "test.csv").write_bytes(b"x" * 500)
        row = ds.status("river_do", with_size=True)[0]
        assert row.size_bytes is not None and row.size_bytes >= 1500

    def test_size_walk_is_capped(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.bin").write_bytes(b"x" * 10)
        capped = ds._dir_size(tmp_path, cap_files=2)  # noqa: SLF001
        assert capped <= 50

    def test_size_text_kilobytes(self):
        # 阈值是 0.1 MB：不到 100 KB 才用 KB 显示。
        row = ds.DatasetStatus(spec=ds.DATASETS["fish_market"], size_bytes=50 * 1024)
        assert row.size_text == "50 KB"

    def test_size_text_megabytes(self):
        row = ds.DatasetStatus(spec=ds.DATASETS["fish_market"], size_bytes=5 * 1024 * 1024)
        assert row.size_text == "5.0 MB"

    def test_available_flag_follows_path(self):
        assert not ds.DatasetStatus(spec=ds.DATASETS["fish_market"]).available
        assert ds.DatasetStatus(spec=ds.DATASETS["fish_market"], path=Path("x")).available


# --------------------------------------------------------------------------- #
# 回归守卫：这次改造的**全部意义**就是干掉硬编码绝对路径
# --------------------------------------------------------------------------- #


class TestNoHardcodedDataPaths:
    MARKER_DIR = "Agent" + "\\" + "数据集"
    MARKER_DIR_POSIX = "Agent/" + "数据集"

    def _offenders(self, root: Path) -> list[str]:
        bad = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if self.MARKER_DIR in text or self.MARKER_DIR_POSIX in text:
                bad.append(str(path.relative_to(ds.PROJECT_ROOT)))
        return bad

    def test_src_has_no_hardcoded_dataset_paths(self):
        assert self._offenders(ds.PROJECT_ROOT / "src") == []

    def test_scripts_have_no_hardcoded_dataset_paths(self):
        assert self._offenders(ds.PROJECT_ROOT / "scripts") == []

    def test_tests_have_no_hardcoded_dataset_paths(self):
        assert self._offenders(ds.PROJECT_ROOT / "tests") == []


# --------------------------------------------------------------------------- #
# 调用方接上了：模块级默认值现在走注册表
# --------------------------------------------------------------------------- #


class TestCallersWiredUp:
    def test_ponds_default_dir_comes_from_the_registry(self):
        from fisheries.evaluation import ponds

        spec = ds.DATASETS["aquaponics_pond"]
        assert ponds.DATASET_NAME == spec.name
        assert ponds.DEFAULT_DIR.name == spec.entry
        assert ponds.DEFAULT_DIR.parent in (ds.LEGACY_ROOT, ds.LOCAL_ROOT) or (
            ponds.DEFAULT_DIR == ds.LOCAL_ROOT / spec.entry
        )

    def test_ponds_defaults_do_not_raise_without_data(self):
        from fisheries.evaluation import ponds

        assert isinstance(ponds.DEFAULT_DIR, Path)

    def test_do_trend_no_longer_resolves_a_dataset_path(self):
        # 第二组验证的数据源改成池塘滑窗后，do_trend 收 by_pond 字典，
        # 不再自己解析数据集路径——数据集入口统一由 ponds 负责。
        from fisheries.evaluation import do_trend

        for gone in ("DEFAULT_DIR", "DEFAULT_TRAIN_CSV", "DEFAULT_TEST_CSV"):
            assert not hasattr(do_trend, gone)
        assert do_trend.DEFAULT_STEP_MINUTES == 10
        assert do_trend.HISTORY_CHOICES == (6, 12, 18)

    def test_evaluation_package_imports_without_any_dataset(self, isolated, monkeypatch):
        # 导入期不该因为缺数据就失败。
        monkeypatch.setattr(ds, "LEGACY_ROOT", ds.PROJECT_ROOT / "nope")
        monkeypatch.setattr(ds, "LOCAL_ROOT", ds.PROJECT_ROOT / "nope2")
        import fisheries.evaluation as ev

        assert ev is not None

    def test_api_sample_probe_uses_the_registry(self):
        text = (
            ds.PROJECT_ROOT / "scripts" / "probes" / "_make_api_samples.py"
        ).read_text(encoding="utf-8")
        assert "datasets" in text
        assert "fish_disease" in text


# --------------------------------------------------------------------------- #
# 探针/演示脚本的「可脱离调用方直接运行」契约
# --------------------------------------------------------------------------- #


class TestScriptsRunStandalone:
    #: 只认真正的导入语句——有的脚本只是路径里带 "fisheries" 字样。
    IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+fisheries", re.MULTILINE)

    def test_scripts_insert_src_on_sys_path(self):
        """任何 import fisheries 的脚本都必须能直接跑。

        它们以前靠调用方先设 PYTHONPATH；直接 `python scripts\\xxx.py` 会
        ModuleNotFoundError。这个坑最初只在探针里发现，后来发现 scripts\\ 根目录
        下也有（demo_cases.py），所以这里查整棵 scripts\\ 树而不是只看 probes\\。
        """
        missing = []
        for path in sorted((ds.PROJECT_ROOT / "scripts").rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if self.IMPORT_RE.search(text) and "sys.path.insert" not in text:
                missing.append(str(path.relative_to(ds.PROJECT_ROOT)))
        assert missing == []

    #: 只在「要求用户先设环境变量」时才算问题；注释里说明「不再需要 PYTHONPATH」
    #: 是好做法，不能一起打掉。
    PYTHONPATH_PRESCRIPTION_RE = re.compile(
        r"(?:\$env:PYTHONPATH\s*=|set\s+PYTHONPATH|PYTHONPATH\s*=\s*[\"'])"
    )

    def test_scripts_do_not_prescribe_a_pythonpath_requirement(self):
        offenders = []
        for path in sorted((ds.PROJECT_ROOT / "scripts").rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if self.PYTHONPATH_PRESCRIPTION_RE.search(text):
                offenders.append(str(path.relative_to(ds.PROJECT_ROOT)))
        assert offenders == []

    def test_scripts_derive_root_from_file_not_from_a_literal(self):
        literal = 'Path(r"D:' + "\\"
        offenders = []
        for path in sorted((ds.PROJECT_ROOT / "scripts").rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if literal in text:
                offenders.append(str(path.relative_to(ds.PROJECT_ROOT)))
        assert offenders == []


# --------------------------------------------------------------------------- #
# 索引文件本身是 JSON，别把中文写坏
# --------------------------------------------------------------------------- #


def test_index_is_written_as_readable_utf8(isolated):
    ds._record("river_do", Path("C:/a/b"))  # noqa: SLF001
    raw = ds.INDEX_PATH.read_text(encoding="utf-8")
    assert json.loads(raw) == {"river_do": str(Path("C:/a/b"))}
    assert raw.endswith("\n")
