#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com

"""utils/log.py 测试。

分两层：
1. 单元测试——库约定不变量（包 logger 不自带真实 handler、默认 propagate）、
   get_logger 无副作用、笔记本 setup 的幂等/级别/propagate。
2. 集成测试（子进程跑 Hydra）——验证 hydra/job_logging 真正生效：app.log 落盘、
   console 仅 WARNING+、多 run 各自独立日志、文件名可覆盖。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable

import pytest

from shifting_baseline.utils.log import (
    PACKAGE_LOGGER_NAME,
    adjust_log_level,
    get_logger,
    setup_logger_from_hydra,
    setup_notebook_logging,
)

# ============================ 单元测试 ============================


def _consoles(logger: logging.Logger) -> list[logging.Handler]:
    return [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]


@pytest.fixture
def clean_package_logger():
    """快照并在测试后恢复包 logger 的全局状态（handlers/level/propagate）。"""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    handlers = logger.handlers[:]
    level, propagate = logger.level, logger.propagate
    yield logger
    for h in logger.handlers:
        if h not in handlers:
            h.close()
    logger.handlers[:] = handlers
    logger.level = level
    logger.propagate = propagate


class TestLibraryConvention:
    """防 Hydra 冲突的核心约定。"""

    def test_only_nullhandler_at_import(self, clean_package_logger):
        # 导入期包根 logger 只能有 NullHandler，绝不能自带 console/file handler，
        # 否则会与 Hydra 配置的 root 重复输出 / 抢同一日志文件。
        real = [
            h
            for h in clean_package_logger.handlers
            if not isinstance(h, logging.NullHandler)
        ]
        assert real == [], f"包 logger 不应自带真实 handler: {real}"

    def test_propagate_true_at_import(self, clean_package_logger):
        # 默认必须 propagate，记录才能上抛到 Hydra 的 root handler。
        assert clean_package_logger.propagate is True


class TestGetLogger:
    def test_default_returns_package_logger(self):
        assert get_logger().name == PACKAGE_LOGGER_NAME

    def test_named_logger(self):
        assert get_logger("shifting_baseline.data").name == "shifting_baseline.data"

    def test_no_side_effects(self, clean_package_logger):
        before = list(clean_package_logger.handlers)
        propagate = clean_package_logger.propagate
        get_logger("shifting_baseline.anything")
        assert list(clean_package_logger.handlers) == before
        assert clean_package_logger.propagate is propagate


class TestNotebookSetup:
    def test_adds_single_console_handler(self, clean_package_logger):
        setup_notebook_logging("INFO")
        assert len(_consoles(clean_package_logger)) == 1

    def test_idempotent(self, clean_package_logger):
        setup_notebook_logging("INFO")
        setup_notebook_logging("WARNING")
        consoles = _consoles(clean_package_logger)
        assert len(consoles) == 1  # 不重复叠加
        assert consoles[0].level == logging.WARNING  # 只调级别

    def test_propagate_disabled(self, clean_package_logger):
        # 笔记本里自己就是 handler，需关 propagate 防重复输出
        setup_notebook_logging("INFO")
        assert clean_package_logger.propagate is False

    def test_console_level(self, clean_package_logger):
        logger = setup_notebook_logging("DEBUG")
        assert _consoles(logger)[0].level == logging.DEBUG

    def test_no_file_by_default(self, clean_package_logger):
        logger = setup_notebook_logging("INFO")
        assert [h for h in logger.handlers if isinstance(h, logging.FileHandler)] == []

    def test_optional_file_handler_writes(self, clean_package_logger, tmp_path):
        fpath = tmp_path / "nb.log"
        logger = setup_notebook_logging(
            "INFO", file_path=str(fpath), file_level="DEBUG"
        )
        logger.debug("hello-debug-to-file")
        files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(files) == 1
        files[0].flush()
        assert "hello-debug-to-file" in fpath.read_text(encoding="utf-8")


class TestAdjustLogLevel:
    def test_lowers_logger_level_for_file(self, clean_package_logger):
        # console=INFO + file=DEBUG → logger 整体级别降到 DEBUG
        adjust_log_level(console_level="INFO", file_level="DEBUG")
        assert clean_package_logger.level == logging.DEBUG

    def test_sets_up_console_when_missing(self, clean_package_logger):
        adjust_log_level(console_level="WARNING")
        consoles = _consoles(clean_package_logger)
        assert len(consoles) == 1
        assert consoles[0].level == logging.WARNING


class TestHydraSetup:
    def test_setup_from_hydra_is_noop(self, clean_package_logger):
        before = list(clean_package_logger.handlers)
        result = setup_logger_from_hydra(cfg=None)
        assert result.name == PACKAGE_LOGGER_NAME
        # Hydra 自己管 root，本函数不应新增任何 handler
        assert list(clean_package_logger.handlers) == before


# ===================== 集成测试（子进程 + Hydra）=====================


def _run_module(
    python_exe: str, cwd: Path, args: Iterable[str]
) -> subprocess.CompletedProcess:
    """Run `python -m shifting_baseline` with given args.

    Args:
        python_exe: Path to Python executable.
        cwd: Working directory to run in.
        args: CLI arguments to pass after the module name.

    Returns:
        CompletedProcess: Result including stdout/stderr for assertions.
    """
    cmd = [python_exe, "-m", "shifting_baseline", *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )


class TestSingleRunLogging:
    """Tests for single-run Hydra logging integration.

    These tests verify that:
    1) A log file is created under the configured run directory.
    2) Console output is WARNING or higher (no INFO/DEBUG).
    3) File log captures DEBUG messages from the application.
    """

    def test_log_file_created_and_levels(
        self, python_bin: str, repo_root: Path, tmp_path: Path
    ) -> None:
        """Single run should create app.log with DEBUG content and console at WARNING."""
        run_dir = tmp_path / "run1"
        cp = _run_module(
            python_bin, repo_root, [f"hydra.run.dir={run_dir}", "test_mode=true"]
        )
        # Console should not contain INFO/DEBUG
        assert " DEBUG " not in cp.stdout
        assert " INFO " not in cp.stdout

        # File logger should exist and contain DEBUG
        log_file = run_dir / "app.log"
        assert log_file.exists(), f"Missing log file: {log_file}"
        content = log_file.read_text(encoding="utf-8")
        assert (
            " DEBUG " in content or "DEBUG" in content
        ), "File log should include DEBUG lines"

        # Verify test log messages are present
        assert "这是一条 DEBUG 日志消息" in content
        assert "这是一条 INFO 日志消息" in content
        assert "这是一条 WARNING 日志消息" in content
        assert "这是一条 ERROR 日志消息" in content
        assert "这是一条 CRITICAL 日志消息" in content


@pytest.mark.parametrize(
    "sweep_param",
    [
        "how.name=process,abm",  # sweep over an existing dimension
    ],
)
class TestMultiRunLogging:
    """Tests for multi-run (sweeps) logging behavior.

    These tests verify that each child job gets its own log under the sweep dir.
    """

    def test_each_job_has_separate_log(
        self, python_bin: str, repo_root: Path, tmp_path: Path, sweep_param: str
    ) -> None:
        """Multirun should create a log file for each job under hydra.sweep.dir."""
        sweep_dir = tmp_path / "sweep"
        cp = _run_module(
            python_bin,
            repo_root,
            [
                "-m",
                f"hydra.sweep.dir={sweep_dir}",
                "hydra.sweep.subdir=${hydra.job.num}",
                "test_mode=true",
                sweep_param,
            ],
        )
        # No INFO/DEBUG on console output
        assert " DEBUG " not in cp.stdout
        assert " INFO " not in cp.stdout

        # Expect two jobs: 0 and 1
        for job_idx in ("0", "1"):
            job_dir = sweep_dir / job_idx
            log_file = job_dir / "app.log"
            assert log_file.exists(), f"Missing log for job {job_idx}: {log_file}"
            content = log_file.read_text(encoding="utf-8")
            assert "DEBUG" in content, f"Job {job_idx} log should include DEBUG lines"

            # Verify specific test log messages are present
            assert "这是一条 DEBUG 日志消息" in content
            assert "日志测试完成，程序退出" in content


class TestOverrideLogFilename:
    """Tests for overriding log filename via Hydra override.

    Ensures that users can change the filename from the default 'app.log'.
    """

    def test_override_log_filename(
        self, python_bin: str, repo_root: Path, tmp_path: Path
    ) -> None:
        """Single run with filename override should write to custom name."""
        run_dir = tmp_path / "run_custom"
        custom_file = run_dir / "custom.log"
        cp = _run_module(
            python_bin,
            repo_root,
            [
                f"hydra.run.dir={run_dir}",
                f"hydra.job_logging.handlers.file.filename={custom_file}",
                "test_mode=true",
            ],
        )
        # Console should not contain INFO/DEBUG
        assert " DEBUG " not in cp.stdout
        assert " INFO " not in cp.stdout

        # Custom file exists, default does not
        assert custom_file.exists(), f"Missing overridden log file: {custom_file}"
        assert not (
            run_dir / "app.log"
        ).exists(), "Default app.log should not be created when overridden"

        # Verify test log messages are present in custom file
        content = custom_file.read_text(encoding="utf-8")
        assert "这是一条 DEBUG 日志消息" in content
        assert "日志测试完成，程序退出" in content


class TestLoggingTestMode:
    """Tests for the test_mode functionality.

    These tests verify that when test_mode=true is set, the application:
    1) Only executes logging tests without business logic
    2) Outputs all log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    3) Exits cleanly without processing data
    """

    def test_test_mode_produces_all_log_levels(
        self, python_bin: str, repo_root: Path, tmp_path: Path
    ) -> None:
        """Test mode should produce all log levels and exit cleanly."""
        run_dir = tmp_path / "test_mode_run"
        cp = _run_module(
            python_bin, repo_root, [f"hydra.run.dir={run_dir}", "test_mode=true"]
        )

        # Should complete successfully without errors
        assert cp.returncode == 0

        # Console should not contain INFO/DEBUG (they go to file)
        assert " DEBUG " not in cp.stdout
        assert " INFO " not in cp.stdout

        # Log file should exist and contain all test messages
        log_file = run_dir / "app.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")

        # Verify all log levels are present
        expected_messages = [
            "这是一条 DEBUG 日志消息",
            "这是一条 INFO 日志消息",
            "这是一条 WARNING 日志消息",
            "这是一条 ERROR 日志消息",
            "这是一条 CRITICAL 日志消息",
            "日志测试完成，程序退出",
        ]

        for message in expected_messages:
            assert message in content, f"Missing expected message: {message}"

    def test_normal_mode_vs_test_mode(
        self, python_bin: str, repo_root: Path, tmp_path: Path
    ) -> None:
        """Compare normal mode vs test mode execution."""
        # Test mode run
        test_run_dir = tmp_path / "test_mode"
        test_cp = _run_module(
            python_bin, repo_root, [f"hydra.run.dir={test_run_dir}", "test_mode=true"]
        )

        # Test mode should complete much faster and successfully
        assert test_cp.returncode == 0

        test_log = test_run_dir / "app.log"
        test_content = test_log.read_text(encoding="utf-8")

        # Test mode should NOT contain business logic messages
        business_messages = [
            "实验开始，配置文件请参看",
            "Step 1: 加载数据",
            "Step 2: 比较每个树轮数据",
            "Step 3: 整合树轮数据",
            "Step 4: 历史数据时期对比",
        ]

        for message in business_messages:
            assert (
                message not in test_content
            ), f"Test mode should not contain: {message}"

        # But should contain test messages
        assert "这是一条 DEBUG 日志消息" in test_content
        assert "日志测试完成，程序退出" in test_content
