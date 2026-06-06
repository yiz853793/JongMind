import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from torch.utils.data import DataLoader

from jongmind.action_space import (
    PASS_ACTION_ID,
    RIICHI_ACTION_ID,
    RON_ACTION_ID,
    human_readable_to_action_id,
)
from jongmind.replay_dataset import MahjongReplayDataset, read_hand_history_jsonl


class ReplayDatasetTest(unittest.TestCase):
    def test_replay_jsonl_builds_samples_without_leaking_hidden_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "hands.jsonl"
            output_path = Path(directory) / "imitation.pt"
            _write_replay(log_path)

            records = list(read_hand_history_jsonl(log_path))
            self.assertEqual(len(records), 1)

            dataset = MahjongReplayDataset(log_path)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                {dataset.raw_sample(index)["sample_type"] for index in range(len(dataset))},
                {"discard", "reaction", "riichi", "win_pass"},
            )

            for index in range(len(dataset)):
                raw = dataset.raw_sample(index)
                item = dataset[index]
                obs = json.loads(item["obs"])
                review = json.loads(item["review"])

                self.assertNotIn("hidden_state_for_review_only", obs)
                self.assertNotIn("hidden_state_for_review_only", raw["obs_public"])
                self.assertNotIn("hidden_state_for_review_only", item)
                self.assertIn("hidden_state_for_review_only", review)
                self.assertIn(raw["action"], raw["legal_actions"])
                self.assertTrue(bool(item["legal_actions"][int(item["action"])]))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jongmind.build_dataset",
                    "--input",
                    str(log_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("saved 4 samples", result.stdout)

            saved_dataset = MahjongReplayDataset(output_path)
            loader = DataLoader(saved_dataset, batch_size=2)
            batch = next(iter(loader))

            self.assertEqual(batch["action"].shape, (2,))
            self.assertEqual(batch["legal_actions"].shape[0], 2)
            self.assertEqual(len(batch["obs"]), 2)


def _write_replay(path: Path) -> None:
    discard_id = human_readable_to_action_id("discard_1m")
    pon_id = human_readable_to_action_id("pon_5m_5m")
    record = {
        "schema": "jongmind.hand_history.v1",
        "hand_index": 0,
        "result": {"type": "ron", "winners": ["SOUTH"], "loser": "WEST"},
        "events": [
            _event(
                action=discard_id,
                action_text="discard_1m",
                legal_actions=[discard_id, RIICHI_ACTION_ID],
                actor="EAST",
                phase="discard",
            ),
            _event(
                action=RIICHI_ACTION_ID,
                action_text="riichi",
                legal_actions=[discard_id, RIICHI_ACTION_ID],
                actor="EAST",
                phase="discard",
                command={"kind": "discard", "seat": "EAST", "tile": "1m", "riichi": True},
            ),
            _event(
                action=pon_id,
                action_text="pon_5m_5m",
                legal_actions=[PASS_ACTION_ID, pon_id],
                actor="SOUTH",
                phase="reaction",
                command={"kind": "call", "seat": "SOUTH", "action": "pon", "tiles": ["5m", "5m"]},
            ),
            _event(
                action=PASS_ACTION_ID,
                action_text="pass",
                legal_actions=[PASS_ACTION_ID, RON_ACTION_ID],
                actor="NORTH",
                phase="reaction",
                command={"kind": "pass", "seat": "NORTH"},
            ),
        ],
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _event(
    action: int,
    action_text: str,
    legal_actions: list[int],
    actor: str,
    phase: str,
    command: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "command",
        "index": len(action_text),
        "actor": actor,
        "phase": phase,
        "command": command or {"kind": "discard", "seat": actor, "tile": "1m"},
        "action": action,
        "action_text": action_text,
        "scores_before": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
        "scores_after": {"EAST": 24000, "SOUTH": 26000, "WEST": 25000, "NORTH": 25000},
        "obs_public": {
            "seat": actor,
            "phase": phase,
            "legal_action_ids": legal_actions,
            "scores": {"EAST": 25000, "SOUTH": 25000, "WEST": 25000, "NORTH": 25000},
            "hidden_state_for_review_only": {"should": "be stripped"},
        },
        "legal_actions": legal_actions,
        "hidden_state_for_review_only": {"hands": {actor: ["1m"]}},
        "result": {"event": action_text},
    }


if __name__ == "__main__":
    unittest.main()
