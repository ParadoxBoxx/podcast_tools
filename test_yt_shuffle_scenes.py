#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yt_shuffle_scenes import (
    EndCardFilterSuppression,
    EndCardFilterSettings,
    TextOnlyFilterSettings,
    TextOnlyFilterSuppression,
    build_end_card_visual_analysis_settings,
    analyze_sampled_frames,
    filter_end_card_segments,
    filter_text_only_segments,
    write_segment_filter_manifest,
    write_summary,
)


def draw_rect(frame: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    for y in range(max(y0, 0), max(y1, 0)):
        row_offset = y * width
        for x in range(max(x0, 0), max(x1, 0)):
            frame[row_offset + x] = value


def build_text_card_frames(width: int, height: int) -> list[bytes]:
    frame = bytearray(width * height)
    draw_rect(frame, width, 16, 18, width - 16, 24, 255)
    draw_rect(frame, width, 28, 30, width - 28, 35, 255)
    draw_rect(frame, width, 22, 42, width - 22, 47, 255)
    return [bytes(frame) for _ in range(3)]


def build_dense_text_card_frames(width: int, height: int) -> list[bytes]:
    frames: list[bytes] = []
    for logo_offset in (0, 1, 0):
        frame = bytearray(width * height)
        draw_rect(frame, width, 10, 8, width - 10, 14, 235)
        draw_rect(frame, width, 18, 20, width - 18, 26, 255)
        draw_rect(frame, width, 24, 31, width - 24, 36, 210)
        draw_rect(frame, width, 20, 42, width - 20, 47, 248)
        draw_rect(frame, width, width - 26 + logo_offset, 8, width - 10 + logo_offset, 22, 180)
        draw_rect(frame, width, width - 24 + logo_offset, 10, width - 12 + logo_offset, 20, 245)
        frames.append(bytes(frame))
    return frames


def build_overlay_footage_frames(width: int, height: int) -> list[bytes]:
    frames: list[bytes] = []
    for offset in (0, 17, 34):
        frame = bytearray(width * height)
        for y in range(height):
            row_offset = y * width
            for x in range(width):
                frame[row_offset + x] = (x * 5 + y * 3 + offset) % 256
        draw_rect(frame, width, 14, height - 20, width - 14, height - 16, 255)
        draw_rect(frame, width, 28, height - 12, width - 28, height - 9, 245)
        frames.append(bytes(frame))
    return frames


class AnalyzeSampledFramesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TextOnlyFilterSettings(
            enabled=True,
            sample_count=3,
            max_motion=4.0,
            min_dominant_coverage=0.72,
            min_edge_density=0.01,
            max_edge_density=0.12,
            max_edge_row_coverage=0.60,
            frame_width=96,
            frame_height=54,
            edge_threshold=18,
        )

    def test_drops_static_text_card_frames(self) -> None:
        analysis = analyze_sampled_frames(build_text_card_frames(96, 54), settings=self.settings)
        self.assertTrue(analysis["drop"])
        self.assertEqual(analysis["reason"], "text_only_heuristic_matched")

    def test_keeps_moving_footage_with_overlay_text(self) -> None:
        analysis = analyze_sampled_frames(build_overlay_footage_frames(96, 54), settings=self.settings)
        self.assertFalse(analysis["drop"])
        self.assertIn("moving_background", str(analysis["reason"]))

    def test_drops_dense_static_title_card(self) -> None:
        analysis = analyze_sampled_frames(build_dense_text_card_frames(96, 54), settings=self.settings)
        self.assertTrue(analysis["drop"])
        self.assertLess(float(analysis["average_edge_row_coverage"]), self.settings.max_edge_row_coverage)


class TextOnlyFilterSuppressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TextOnlyFilterSettings(
            enabled=True,
            sample_count=3,
            max_motion=4.0,
            min_dominant_coverage=0.72,
            min_edge_density=0.01,
            max_edge_density=0.12,
            max_edge_row_coverage=0.60,
            frame_width=96,
            frame_height=54,
            edge_threshold=18,
        )
        self.end_card_settings = EndCardFilterSettings(
            enabled=True,
            window_seconds=18.0,
            min_start_ratio=0.82,
            min_duration=1.0,
            max_duration=20.0,
            max_motion=2.2,
            min_dominant_coverage=0.42,
            min_edge_density=0.008,
            max_edge_density=0.26,
            max_edge_row_coverage=0.82,
        )
        self.segments = [
            {"scene_index": 0, "start": 0.0, "end": 1.0, "duration": 1.0},
            {"scene_index": 1, "start": 1.0, "end": 2.0, "duration": 1.0},
        ]

    def test_preserves_original_flagged_classification_when_suppression_fires(self) -> None:
        classifications = [
            {"enabled": True, "drop": True, "reason": "text_only_heuristic_matched"},
            {"enabled": True, "drop": True, "reason": "text_only_heuristic_matched"},
        ]
        with patch("yt_shuffle_scenes.classify_segment_for_text_only_filter", side_effect=classifications):
            kept, dropped, candidates, suppression = filter_text_only_segments(
                Path("/tmp/source.mp4"),
                self.segments,
                self.settings,
            )

        self.assertTrue(suppression.fired)
        self.assertEqual(suppression.reason, "all_segments_flagged")
        self.assertEqual(suppression.suppressed_drop_count, 2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 0)
        self.assertTrue(all(segment["text_only_filter"]["drop"] for segment in candidates))
        self.assertTrue(all(segment["text_only_filter"]["drop"] for segment in kept))
        self.assertTrue(all(not segment["text_only_filter_effective_drop"] for segment in candidates))
        self.assertTrue(all(not segment["text_only_filter_effective_drop"] for segment in kept))

    def test_reports_suppression_without_erasing_flagged_segments(self) -> None:
        suppression = TextOnlyFilterSuppression(
            fired=True,
            reason="all_segments_flagged",
            suppressed_drop_count=2,
            message="Text-only filter suppression fired because every detected segment was flagged; preserving all segments for this run.",
        )
        end_card_suppression = EndCardFilterSuppression(fired=False)
        candidates = [
            {
                "scene_index": 0,
                "source_scene_index": 0,
                "start": 0.0,
                "end": 1.0,
                "duration": 1.0,
                "text_only_filter": {"enabled": True, "drop": True, "reason": "text_only_heuristic_matched"},
                "text_only_filter_effective_drop": False,
            },
            {
                "scene_index": 1,
                "source_scene_index": 1,
                "start": 1.0,
                "end": 2.0,
                "duration": 1.0,
                "text_only_filter": {"enabled": True, "drop": True, "reason": "text_only_heuristic_matched"},
                "text_only_filter_effective_drop": False,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.json"
            report_path = root / "report.json"
            write_summary(
                summary_path,
                source="demo.mp4",
                formatted_path=root / "formatted.mp4",
                final_output_path=root / "final.mp4",
                filter_manifest_path=report_path,
                scene_times=[0.5],
                segments=candidates,
                dropped_segments=[],
                flagged_segments=candidates,
                candidate_segment_count=len(candidates),
                shuffled_order=[1, 0],
                shuffle_seed=0,
                encoder_used="copy",
                text_only_filter_settings=self.settings,
                text_only_filter_suppression=suppression,
                end_card_filter_settings=self.end_card_settings,
                end_card_filter_suppression=end_card_suppression,
            )
            write_segment_filter_manifest(
                report_path,
                text_only_filter_settings=self.settings,
                end_card_filter_settings=self.end_card_settings,
                segments=candidates,
                text_only_filter_suppression=suppression,
                end_card_filter_suppression=end_card_suppression,
            )

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(summary_payload["text_only_filter"]["suppression"]["fired"])
        self.assertEqual(summary_payload["text_only_filter"]["flagged_segment_count"], 2)
        self.assertEqual(summary_payload["text_only_filter"]["dropped_segment_count"], 0)
        self.assertEqual(len(summary_payload["flagged_segments"]), 2)
        self.assertEqual(report_payload["text_only_filter"]["flagged_segment_count"], 2)
        self.assertEqual(report_payload["text_only_filter"]["effective_dropped_segment_count"], 0)
        self.assertTrue(report_payload["text_only_filter"]["suppression"]["fired"])


class EndCardFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = EndCardFilterSettings(
            enabled=True,
            window_seconds=18.0,
            min_start_ratio=0.82,
            min_duration=1.0,
            max_duration=20.0,
            max_motion=2.2,
            min_dominant_coverage=0.42,
            min_edge_density=0.008,
            max_edge_density=0.26,
            max_edge_row_coverage=0.82,
        )
        self.text_settings = TextOnlyFilterSettings(
            enabled=True,
            sample_count=3,
            max_motion=4.0,
            min_dominant_coverage=0.68,
            min_edge_density=0.01,
            max_edge_density=0.16,
            max_edge_row_coverage=0.60,
            frame_width=96,
            frame_height=54,
            edge_threshold=18,
        )
        self.end_card_visual_settings = build_end_card_visual_analysis_settings(self.text_settings, self.settings)
        self.no_suppression = TextOnlyFilterSuppression(fired=False)

    def test_removes_only_trailing_late_end_card_suffix(self) -> None:
        segments = [
            {
                "scene_index": 0,
                "source_scene_index": 0,
                "start": 0.0,
                "end": 2.5,
                "duration": 2.5,
                "text_only_filter": {
                    "enabled": True,
                    "drop": False,
                    "reason": "moving_background",
                    "motion_score": 12.0,
                    "average_dominant_coverage": 0.31,
                    "average_edge_density": 0.23,
                    "average_edge_row_coverage": 0.88,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
            {
                "scene_index": 1,
                "source_scene_index": 1,
                "start": 2.5,
                "end": 4.6,
                "duration": 2.1,
                "text_only_filter": {
                    "enabled": True,
                    "drop": False,
                    "reason": "moving_background",
                    "motion_score": 6.5,
                    "average_dominant_coverage": 0.33,
                    "average_edge_density": 0.21,
                    "average_edge_row_coverage": 0.77,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
            {
                "scene_index": 2,
                "source_scene_index": 2,
                "start": 4.6,
                "end": 6.0,
                "duration": 1.4,
                "text_only_filter": {
                    "enabled": True,
                    "drop": False,
                    "reason": "kept_ambiguous",
                    "motion_score": 0.9,
                    "average_dominant_coverage": 0.73,
                    "average_edge_density": 0.11,
                    "average_edge_row_coverage": 0.42,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
        ]

        kept, dropped, suppression = filter_end_card_segments(
            Path("/tmp/source.mp4"),
            segments,
            total_duration=6.0,
            settings=self.settings,
            visual_analysis_settings=self.end_card_visual_settings,
            text_filter_suppression=self.no_suppression,
        )

        self.assertFalse(suppression.fired)
        self.assertEqual([segment["source_scene_index"] for segment in kept], [0, 1])
        self.assertEqual([segment["source_scene_index"] for segment in dropped], [2])
        self.assertEqual(dropped[0]["end_card_filter"]["reason"], "late_trailing_branded_end_card")
        self.assertIn("end_card", dropped[0]["removal_reasons"])
        self.assertFalse(kept[1]["end_card_filter_effective_drop"])
        self.assertIn("not_trailing_suffix", str(kept[0]["end_card_filter"]["reason"]))

    def test_runs_independent_visual_analysis_when_text_filter_is_disabled(self) -> None:
        segments = [
            {
                "scene_index": 0,
                "source_scene_index": 0,
                "start": 0.0,
                "end": 4.6,
                "duration": 4.6,
                "text_only_filter": {
                    "enabled": True,
                    "drop": False,
                    "reason": "moving_background",
                    "motion_score": 9.0,
                    "average_dominant_coverage": 0.28,
                    "average_edge_density": 0.22,
                    "average_edge_row_coverage": 0.8,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
            {
                "scene_index": 1,
                "source_scene_index": 1,
                "start": 4.6,
                "end": 6.0,
                "duration": 1.4,
                "text_only_filter": {"enabled": False, "drop": False, "reason": "disabled"},
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
        ]

        with patch("yt_shuffle_scenes.sample_segment_frames", return_value=build_dense_text_card_frames(96, 54)):
            kept, dropped, suppression = filter_end_card_segments(
                Path("/tmp/source.mp4"),
                segments,
                total_duration=6.0,
                settings=self.settings,
                visual_analysis_settings=self.end_card_visual_settings,
                text_filter_suppression=self.no_suppression,
            )

        self.assertFalse(suppression.fired)
        self.assertEqual([segment["source_scene_index"] for segment in kept], [0])
        self.assertEqual([segment["source_scene_index"] for segment in dropped], [1])
        self.assertEqual(dropped[0]["end_card_filter"]["analysis_source"], "independent_end_card_analysis")
        self.assertTrue(dropped[0]["end_card_filter_effective_drop"])

    def test_preserves_segments_when_end_card_filter_would_remove_everything(self) -> None:
        segments = [
            {
                "scene_index": 0,
                "source_scene_index": 0,
                "start": 4.2,
                "end": 5.1,
                "duration": 0.9,
                "text_only_filter": {
                    "enabled": True,
                    "drop": True,
                    "reason": "text_only_heuristic_matched",
                    "motion_score": 0.6,
                    "average_dominant_coverage": 0.78,
                    "average_edge_density": 0.09,
                    "average_edge_row_coverage": 0.33,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
            {
                "scene_index": 1,
                "source_scene_index": 1,
                "start": 5.1,
                "end": 6.0,
                "duration": 0.9,
                "text_only_filter": {
                    "enabled": True,
                    "drop": True,
                    "reason": "text_only_heuristic_matched",
                    "motion_score": 0.4,
                    "average_dominant_coverage": 0.8,
                    "average_edge_density": 0.1,
                    "average_edge_row_coverage": 0.3,
                },
                "text_only_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
        ]

        preserved_settings = EndCardFilterSettings(
            enabled=True,
            window_seconds=18.0,
            min_start_ratio=0.60,
            min_duration=0.5,
            max_duration=20.0,
            max_motion=2.2,
            min_dominant_coverage=0.42,
            min_edge_density=0.008,
            max_edge_density=0.26,
            max_edge_row_coverage=0.82,
        )

        kept, dropped, suppression = filter_end_card_segments(
            Path("/tmp/source.mp4"),
            segments,
            total_duration=6.0,
            settings=preserved_settings,
            visual_analysis_settings=self.end_card_visual_settings,
            text_filter_suppression=self.no_suppression,
        )

        self.assertTrue(suppression.fired)
        self.assertEqual(suppression.reason, "all_segments_flagged")
        self.assertEqual(suppression.suppressed_drop_count, 2)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 0)
        self.assertTrue(all(segment["end_card_filter"]["drop"] for segment in kept))
        self.assertTrue(all(not segment["end_card_filter_effective_drop"] for segment in kept))
        self.assertTrue(all(not segment["effective_removed"] for segment in kept))

    def test_summary_and_report_include_end_card_counts(self) -> None:
        suppression = TextOnlyFilterSuppression(fired=False)
        end_card_suppression = EndCardFilterSuppression(fired=False)
        candidates = [
            {
                "scene_index": 0,
                "source_scene_index": 0,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "text_only_filter": {"enabled": True, "drop": False, "reason": "moving_background"},
                "text_only_filter_effective_drop": False,
                "end_card_filter": {"enabled": True, "drop": False, "reason": "not_trailing_suffix"},
                "end_card_filter_effective_drop": False,
                "removal_reasons": [],
                "effective_removed": False,
            },
            {
                "scene_index": 1,
                "source_scene_index": 1,
                "start": 2.0,
                "end": 3.4,
                "duration": 1.4,
                "text_only_filter": {"enabled": True, "drop": False, "reason": "kept_ambiguous"},
                "text_only_filter_effective_drop": False,
                "end_card_filter": {"enabled": True, "drop": True, "reason": "late_trailing_branded_end_card"},
                "end_card_filter_effective_drop": True,
                "removal_reasons": ["end_card"],
                "effective_removed": True,
            },
        ]
        kept_segments = [candidates[0]]
        dropped_segments = [candidates[1]]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.json"
            report_path = root / "report.json"
            write_summary(
                summary_path,
                source="demo.mp4",
                formatted_path=root / "formatted.mp4",
                final_output_path=root / "final.mp4",
                filter_manifest_path=report_path,
                scene_times=[0.5],
                segments=kept_segments,
                dropped_segments=dropped_segments,
                flagged_segments=[],
                candidate_segment_count=len(candidates),
                shuffled_order=[0],
                shuffle_seed=0,
                encoder_used="copy",
                text_only_filter_settings=self.text_settings,
                text_only_filter_suppression=suppression,
                end_card_filter_settings=self.settings,
                end_card_filter_suppression=end_card_suppression,
            )
            write_segment_filter_manifest(
                report_path,
                text_only_filter_settings=self.text_settings,
                end_card_filter_settings=self.settings,
                segments=candidates,
                text_only_filter_suppression=suppression,
                end_card_filter_suppression=end_card_suppression,
            )

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(summary_payload["end_card_filter"]["dropped_segment_count"], 1)
        self.assertEqual(summary_payload["end_card_filter"]["flagged_segment_count"], 1)
        self.assertFalse(summary_payload["end_card_filter"]["suppression"]["fired"])
        self.assertEqual(summary_payload["dropped_segments"][0]["removal_reasons"], ["end_card"])
        self.assertEqual(report_payload["end_card_filter"]["flagged_segment_count"], 1)
        self.assertEqual(report_payload["end_card_filter"]["effective_dropped_segment_count"], 1)
        self.assertFalse(report_payload["end_card_filter"]["suppression"]["fired"])


if __name__ == "__main__":
    unittest.main()