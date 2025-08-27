#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import subprocess
from pathlib import Path
from typing import Iterable

import pytest


def _run_module(
    python_exe: str, cwd: Path, args: Iterable[str]
) -> subprocess.CompletedProcess:
    """Run `python -m past1000` with given args.

    Args:
        python_exe: Path to Python executable.
        cwd: Working directory to run in.
        args: CLI arguments to pass after the module name.

    Returns:
        CompletedProcess: Result including stdout/stderr for assertions.
    """
    cmd = [python_exe, "-m", "past1000", *args]
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
        """Single run should create app.log with DEBUG content and console at WARNING.

        Steps:
        - Enable test_mode to avoid executing business logic.
        - Force hydra.run.dir to a temp directory.
        - Run module once.
        - Assert app.log exists and contains a DEBUG line.
        - Assert stdout has no INFO/DEBUG text.
        """
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
        """Multirun should create a log file for each job under hydra.sweep.dir.

        Steps:
        - Enable test_mode to avoid executing business logic.
        - Use -m to enable multirun and sweep an existing option.
        - Override hydra.sweep.dir and subdir to controlled temp locations.
        - Assert logs exist under each job subdir and contain DEBUG lines.
        """
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
        """Single run with filename override should write to custom name.

        Steps:
        - Enable test_mode to avoid executing business logic.
        - Override job_logging file handler filename to custom.log.
        - Assert 'custom.log' exists and default 'app.log' does not.
        """
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
