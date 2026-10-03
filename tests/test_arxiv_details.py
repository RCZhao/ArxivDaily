"""Offline regression tests for daily arXiv metadata fetching.

Run with: python -m unittest discover -s tests -v
Only arxiv (and its requests dependency) is required. Load the two daily
fetching methods from the source AST to avoid unrelated model imports,
NLTK downloads and Zotero/LLM initialization during unit tests.
"""
import ast
import contextlib
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import re
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import arxiv
import requests


REAL_CLIENT = arxiv.Client
SOURCE = Path(__file__).resolve().parents[1] / "arxiv_engine.py"


def load_daily_engine(namespace):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    engine = next(node for node in source.body
                  if isinstance(node, ast.ClassDef) and node.name == "ArxivEngine")
    methods = [node for node in engine.body
               if isinstance(node, ast.FunctionDef)
               and node.name in {"_fetch_paper_details", "get_recommendations"}]
    assert len(methods) == 2
    isolated = ast.Module(body=[ast.ClassDef(
        name="ArxivEngine", bases=[], keywords=[], body=methods, decorator_list=[],
    )], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(SOURCE), "exec"), namespace)
    return namespace["ArxivEngine"]()


def paper(paper_id, date_offset=0):
    return SimpleNamespace(
        get_short_id=lambda: paper_id + "v1",
        published=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=date_offset),
        score=1.0,
    )


class DailyDetailsTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.client = Mock()
        self.convert = Mock(side_effect=lambda result: result)
        self.feedparser = SimpleNamespace(parse=Mock())
        namespace = {
            "arxiv": arxiv, "requests": requests, "re": re, "time": time,
            "ArxivPaper": self.convert,
            "tqdm": lambda iterable, **kwargs: iterable,
            "feedparser": self.feedparser,
            "ARXIV_CATEGORIES": ["astro-ph.CO"], "TLDR_GENERATOR": "sumy",
        }
        self.engine = load_daily_engine(namespace)
        self.factory = self.stack.enter_context(patch.object(arxiv, "Client", return_value=self.client))
        self.sleep = self.stack.enter_context(patch.object(time, "sleep"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_314_ids_are_split_and_global_date_order_is_restored(self):
        ids = [f"2609.{i:05d}" for i in range(314)]
        results = {paper_id: paper(paper_id, i) for i, paper_id in enumerate(ids)}
        self.client.results.side_effect = lambda search: iter(
            results[paper_id] for paper_id in reversed(search.id_list)
        )
        fetched = self.engine._fetch_paper_details(ids)
        searches = [call.args[0] for call in self.client.results.call_args_list]
        self.assertEqual([len(search.id_list) for search in searches], [100, 100, 100, 14])
        self.assertEqual([item for search in searches for item in search.id_list], ids)
        self.assertEqual([search.max_results for search in searches], [100, 100, 100, 14])
        self.assertTrue(all(search.sort_by == arxiv.SortCriterion.SubmittedDate for search in searches))
        self.assertEqual(fetched, [results[paper_id] for paper_id in reversed(ids)])
        self.factory.assert_called_once_with(page_size=100, delay_seconds=10, num_retries=0)
        self.sleep.assert_not_called()
        self.assertEqual(self.convert.call_count, 314)

        # Exercise the real library's URL formatter, not only Search objects.
        formatter = REAL_CLIENT(page_size=100)
        for search in searches:
            url = formatter._format_url(search, 0, 100)
            encoded_ids = parse_qs(urlparse(url).query)["id_list"][0].split(",")
            self.assertEqual(encoded_ids, search.id_list)
            self.assertLess(len(url), 1500)

    def test_empty_input_does_not_create_client(self):
        self.assertEqual(self.engine._fetch_paper_details([]), [])
        self.factory.assert_not_called()

    def test_duplicate_rss_ids_are_fetched_once(self):
        result = paper("2609.00001")
        self.client.results.return_value = iter([result])
        self.assertEqual(self.engine._fetch_paper_details(["2609.00001"] * 2), [result])
        self.assertEqual(self.client.results.call_args.args[0].id_list, ["2609.00001"])

    def test_429_then_503_recovers_with_backoff(self):
        result = paper("2609.00001")
        self.client.results.side_effect = [
            arxiv.HTTPError("mock://arxiv", 0, 429),
            arxiv.HTTPError("mock://arxiv", 0, 503),
            iter([result]),
        ]
        self.assertEqual(self.engine._fetch_paper_details(["2609.00001"]), [result])
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [30, 60])

    def test_persistent_503_stops_after_four_attempts(self):
        error = arxiv.HTTPError("mock://arxiv", 0, 503)
        self.client.results.side_effect = error
        with self.assertRaises(arxiv.HTTPError) as caught:
            self.engine._fetch_paper_details(["2609.00001"])
        self.assertIs(caught.exception, error)
        self.assertEqual(self.client.results.call_count, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [30, 60, 120])
        self.convert.assert_not_called()

    def test_400_is_not_retried(self):
        self.client.results.side_effect = arxiv.HTTPError("mock://arxiv", 0, 400)
        with self.assertRaises(arxiv.HTTPError):
            self.engine._fetch_paper_details(["2609.00001"])
        self.assertEqual(self.client.results.call_count, 1)
        self.sleep.assert_not_called()

    def test_connection_error_is_retried(self):
        result = paper("2609.00001")
        self.client.results.side_effect = [requests.exceptions.ConnectionError("offline"), iter([result])]
        self.assertEqual(self.engine._fetch_paper_details(["2609.00001"]), [result])
        self.sleep.assert_called_once_with(30)

    def test_non_network_error_is_not_retried(self):
        self.client.results.side_effect = ValueError("invalid metadata")
        with self.assertRaises(ValueError):
            self.engine._fetch_paper_details(["2609.00001"])
        self.sleep.assert_not_called()

    def test_partial_iteration_is_discarded_before_retry(self):
        first, second = paper("2609.00001"), paper("2609.00002")

        def interrupted():
            yield first
            raise arxiv.HTTPError("mock://arxiv", 0, 503)

        self.client.results.side_effect = [interrupted(), iter([first, second])]
        self.assertEqual(self.engine._fetch_paper_details(["2609.00001", "2609.00002"]), [first, second])
        self.assertEqual(self.convert.call_count, 2)

    def test_missing_duplicate_and_unexpected_results_fail_closed(self):
        first = paper("2609.00001")
        for returned in ([], [first], [first, first], [first, paper("2609.99999")]):
            with self.subTest(returned=returned):
                self.client.results.return_value = iter(returned)
                with self.assertRaisesRegex(RuntimeError, "Incomplete arXiv detail batch"):
                    self.engine._fetch_paper_details(["2609.00001", "2609.00002"])
        self.convert.assert_not_called()
        self.sleep.assert_not_called()

    def test_failure_in_later_batch_does_not_return_partial_papers(self):
        ids = [f"2609.{i:05d}" for i in range(101)]
        self.client.results.side_effect = [
            iter([paper(paper_id) for paper_id in ids[:100]]),
            arxiv.HTTPError("mock://arxiv", 0, 400),
        ]
        with self.assertRaises(arxiv.HTTPError):
            self.engine._fetch_paper_details(ids)
        self.convert.assert_not_called()

    def test_daily_path_passes_complete_details_to_scoring(self):
        result = paper("2609.00001")
        self.feedparser.parse.return_value = SimpleNamespace(
            bozo=False, feed=SimpleNamespace(title="arXiv"), entries=[SimpleNamespace(
                arxiv_announce_type="new", id="oai:arXiv.org:2609.00001v1",
            )],
        )
        self.client.results.return_value = iter([result])
        self.engine._score_papers_batch = Mock(return_value=[result])
        recommended, scored = self.engine.get_recommendations(0, 0.0)
        self.assertEqual((recommended, scored), ([], [result]))
        self.engine._score_papers_batch.assert_called_once_with([result], algorithm="z_score")

    def test_daily_path_does_not_score_after_fetch_failure(self):
        self.feedparser.parse.return_value = SimpleNamespace(
            bozo=False, feed=SimpleNamespace(title="arXiv"), entries=[SimpleNamespace(
                arxiv_announce_type="new", id="oai:arXiv.org:2609.00001v1",
            )],
        )
        self.client.results.side_effect = arxiv.HTTPError("mock://arxiv", 0, 400)
        self.engine._score_papers_batch = Mock()
        with self.assertRaises(arxiv.HTTPError):
            self.engine.get_recommendations(10, 0.0)
        self.engine._score_papers_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
