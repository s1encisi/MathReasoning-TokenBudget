"""Offline regressions for provider failures, retry limits, and asynchronous file preparation."""

import asyncio
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from open_r1 import i18n


class SDKFailure(Exception):
    """Stand-in for a provider-specific error without depending on its SDK."""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CodeExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        previous_locale = i18n.get_locale()
        i18n.set_locale("en_US")
        self.addCleanup(i18n.set_locale, previous_locale)
        source = Path(__file__).resolve().parents[1] / "src" / "open_r1" / "utils"
        # Isolate optional training/SDK imports; load the actual implementation files.
        utils = types.ModuleType("open_r1.utils")
        utils.__path__ = [str(source)]
        utils.is_e2b_available = lambda: False
        utils.is_morph_available = lambda: False
        import_utils = types.ModuleType("open_r1.utils.import_utils")
        import_utils.is_morph_available = lambda: False
        modules = patch.dict(sys.modules, {"open_r1.utils": utils, "open_r1.utils.import_utils": import_utils})
        modules.start()
        self.addCleanup(modules.stop)
        self.providers = load_module("open_r1.utils.code_providers", source / "code_providers.py")
        self.morph = load_module(
            "open_r1.utils.competitive_programming.morph_client",
            source / "competitive_programming" / "morph_client.py",
        )

    async def test_e2b_creation_failure_keeps_zero_reward(self):
        provider = self.providers.E2BProvider.__new__(self.providers.E2BProvider)
        sandbox_api = types.SimpleNamespace(create=AsyncMock(side_effect=SDKFailure("creation failed")))
        with (
            patch.object(self.providers, "AsyncSandbox", sandbox_api),
            self.assertLogs(self.providers.logger, level="WARNING") as logs,
        ):
            reward = await provider._run_script("print(1)", ["python"], asyncio.Semaphore(1))

        self.assertEqual(reward, 0.0)
        self.assertIs(logs.records[0].exc_info[0], SDKFailure)

    async def test_e2b_cleanup_failure_preserves_execution_reward(self):
        provider = self.providers.E2BProvider.__new__(self.providers.E2BProvider)
        sandbox = types.SimpleNamespace(
            sandbox_id="offline-test",
            run_code=AsyncMock(return_value=types.SimpleNamespace(text="0.75")),
            kill=AsyncMock(side_effect=SDKFailure("cleanup failed")),
        )
        sandbox_api = types.SimpleNamespace(create=AsyncMock(return_value=sandbox))
        with (
            patch.object(self.providers, "AsyncSandbox", sandbox_api),
            self.assertLogs(self.providers.logger, level="WARNING") as logs,
        ):
            reward = await provider._run_script("print(0.75)", ["python"], asyncio.Semaphore(1))

        self.assertEqual(reward, 0.75)
        sandbox.kill.assert_awaited_once()
        self.assertIs(logs.records[0].exc_info[0], SDKFailure)

    async def test_morph_start_timeout_preserves_original_exception(self):
        client = self.morph.MorphCloudExecutionClient.__new__(self.morph.MorphCloudExecutionClient)
        timeout = asyncio.TimeoutError("start failed")
        client.client = types.SimpleNamespace(instances=types.SimpleNamespace(astart=AsyncMock(side_effect=timeout)))

        with self.assertRaises(asyncio.TimeoutError) as raised:
            await client._prepare_instance(snapshot_id="offline-snapshot")

        self.assertIs(raised.exception, timeout)

    async def test_morph_file_writes_run_outside_event_loop_thread(self):
        client = self.morph.MorphCloudExecutionClient.__new__(self.morph.MorphCloudExecutionClient)
        loop_thread = threading.get_ident()
        write_threads = []
        write_text = Path.write_text

        def record_write(path, content, **kwargs):
            write_threads.append(threading.get_ident())
            return write_text(path, content, **kwargs)

        data = {
            "files": [
                {"name": "graders/example.cpp", "content": "// 中文程序\n"},
                {"name": "input.txt", "content": "1\n"},
            ],
            "run_timeout": 2000,
            "run_memory_limit": 64,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(Path, "write_text", record_write):
                problem_id, config, files = await client._prepare_files(data, temp_dir)

            self.assertEqual(problem_id, "example")
            self.assertEqual(config["time_limit"], 2)
            self.assertEqual(json.loads(Path(files["grader_config.json"]).read_text(encoding="utf-8")), config)
            self.assertEqual(Path(files["graders/example.cpp"]).read_text(encoding="utf-8"), "// 中文程序\n")
        self.assertEqual(len(write_threads), 3)
        self.assertTrue(all(thread != loop_thread for thread in write_threads))

    async def test_morph_sdk_failures_keep_bounded_retries_and_feedback(self):
        client = self.morph.MorphCloudExecutionClient.__new__(self.morph.MorphCloudExecutionClient)
        client._execute = AsyncMock(side_effect=SDKFailure("backend unavailable"))
        with (
            patch.object(self.morph.asyncio, "sleep", new_callable=AsyncMock) as sleep,
            self.assertLogs(self.morph.logger, level="WARNING") as logs,
        ):
            score, feedback = await client.execute({})

        self.assertEqual(score, "0")
        self.assertIn("backend unavailable", feedback)
        self.assertEqual(client._execute.await_count, 5)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.0, 2.0, 4.0, 8.0])
        self.assertEqual(len(logs.records), 5)
        self.assertTrue(all(record.exc_info[0] is SDKFailure for record in logs.records))


if __name__ == "__main__":
    unittest.main()
