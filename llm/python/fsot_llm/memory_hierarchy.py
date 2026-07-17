"""
FSOT memory hierarchy — long-term / short-term / process.

Maps mind–body–soul style cognition onto the observer stack without
loading the whole knowledge fluid into GPU VRAM:

| Layer              | Substrate        | FSOT role                                      |
|--------------------|------------------|------------------------------------------------|
| Long-term (LTM)    | Disk SQLite DB   | Persistent waveform of known information       |
| Short-term (STM)   | System RAM       | Active working set + recent_hits               |
| Process (immediate)| GPU and/or CPU   | Forward pass only — active pathway fold        |

Routing still uses domain map (D_eff, pathway_key, emission).
LTM retrieval is by fold + keyword/tag — not free-parameter embeddings
as the ontology (optional FTS later).

Default LTM path: D:\\FSOT_LLM_Memory\\fsot_ltm.sqlite (external drive)
Fallback: llm/data/memory/fsot_ltm.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

from fsot_llm.domain_routing import DomainAllocation, resolve_allocation
from fsot_llm.paths import ensure_sys_path, workspace_root

ensure_sys_path()


def default_ltm_path() -> Path:
    env = os.environ.get("FSOT_LTM_PATH")
    if env:
        return Path(env)
    d = Path(r"D:\FSOT_LLM_Memory")
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d / "fsot_ltm.sqlite"
    except Exception:
        local = workspace_root() / "llm" / "data" / "memory"
        local.mkdir(parents=True, exist_ok=True)
        return local / "fsot_ltm.sqlite"


# ---------------------------------------------------------------------------
# Long-term memory — disk database (grabable, not VRAM-resident)
# ---------------------------------------------------------------------------
@dataclass
class LTMRecord:
    id: Optional[int]
    content: str
    pathway: str
    domain: str
    D_eff: float
    emission: str
    tags: list[str]
    source: str
    created_at: str
    strength: float = 1.0  # suction residual — decays if unused


class LongTermMemory:
    """
    SQLite-backed episodic/semantic store.
    Never loads full corpus onto GPU — only retrieved slices enter STM.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or default_ltm_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                pathway TEXT NOT NULL,
                domain TEXT NOT NULL,
                D_eff REAL NOT NULL,
                emission TEXT NOT NULL,
                tags TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS idx_ep_pathway ON episodes(pathway);
            CREATE INDEX IF NOT EXISTS idx_ep_domain ON episodes(domain);
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(content, tags, content='episodes', content_rowid='id');
            """
        )
        self._conn.commit()

    def store(
        self,
        content: str,
        *,
        pathway: str = "ontology",
        domain: str = "consciousness",
        D_eff: float = 16.0,
        emission: str = "free",
        tags: Optional[list[str]] = None,
        source: str = "session",
        strength: float = 1.0,
    ) -> int:
        tags = tags or []
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO episodes
            (content, pathway, domain, D_eff, emission, tags, source, created_at, strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                pathway,
                domain,
                float(D_eff),
                emission,
                json.dumps(tags),
                source,
                now,
                strength,
            ),
        )
        rid = int(cur.lastrowid)
        try:
            self._conn.execute(
                "INSERT INTO episodes_fts(rowid, content, tags) VALUES (?, ?, ?)",
                (rid, content, " ".join(tags)),
            )
        except sqlite3.OperationalError:
            pass
        self._conn.commit()
        return rid

    def retrieve(
        self,
        *,
        query: str = "",
        pathway: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: int = 5,
        min_strength: float = 0.05,
    ) -> list[LTMRecord]:
        """Pull LTM slices into RAM (for STM). Does not touch GPU."""
        rows: list[sqlite3.Row] = []
        if query.strip():
            try:
                q = " ".join(query.strip().split()[:12])
                sql = """
                SELECT e.* FROM episodes e
                JOIN episodes_fts f ON e.id = f.rowid
                WHERE episodes_fts MATCH ?
                """
                params: list[Any] = [q]
                if pathway:
                    sql += " AND e.pathway = ?"
                    params.append(pathway)
                if domain:
                    sql += " AND e.domain = ?"
                    params.append(domain)
                sql += " AND e.strength >= ? ORDER BY e.strength DESC LIMIT ?"
                params.extend([min_strength, top_k])
                rows = list(self._conn.execute(sql, params))
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            sql = "SELECT * FROM episodes WHERE strength >= ?"
            params = [min_strength]
            if pathway:
                sql += " AND pathway = ?"
                params.append(pathway)
            if domain:
                sql += " AND domain = ?"
                params.append(domain)
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(top_k)
            rows = list(self._conn.execute(sql, params))

        out: list[LTMRecord] = []
        for r in rows:
            # suction residual: bump strength on retrieval
            self._conn.execute(
                "UPDATE episodes SET strength = MIN(2.0, strength + 0.05) WHERE id = ?",
                (r["id"],),
            )
            out.append(
                LTMRecord(
                    id=r["id"],
                    content=r["content"],
                    pathway=r["pathway"],
                    domain=r["domain"],
                    D_eff=float(r["D_eff"]),
                    emission=r["emission"],
                    tags=json.loads(r["tags"] or "[]"),
                    source=r["source"],
                    created_at=r["created_at"],
                    strength=float(r["strength"]),
                )
            )
        self._conn.commit()
        return out

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def decay(self, factor: float = 0.99) -> None:
        """Poof weak unused memories slowly (yin–yang: suction/poof)."""
        self._conn.execute(
            "UPDATE episodes SET strength = strength * ?",
            (factor,),
        )
        self._conn.execute("DELETE FROM episodes WHERE strength < 0.02")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Short-term memory — RAM only
# ---------------------------------------------------------------------------
@dataclass
class STMTurn:
    role: str
    content: str
    pathway: str
    ts: float = field(default_factory=time.time)


class ShortTermMemory:
    """
    Working set in RAM: recent dialogue + LTM slices pulled this session.
    Never written to GPU weight tensors — only text context for process.
    """

    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self.turns: Deque[STMTurn] = deque(maxlen=capacity)
        self.active_ltm: list[LTMRecord] = []
        self.route: Optional[DomainAllocation] = None
        self.recent_hits: float = 0.0

    def push(self, role: str, content: str, pathway: str = "ontology") -> None:
        self.turns.append(STMTurn(role=role, content=content, pathway=pathway))

    def set_route(self, alloc: DomainAllocation) -> None:
        self.route = alloc

    def load_ltm_slice(self, records: list[LTMRecord]) -> None:
        self.active_ltm = records
        self.recent_hits = min(1.0, self.recent_hits + 0.1 * len(records))

    def clear_working(self) -> None:
        self.turns.clear()
        self.active_ltm = []
        self.recent_hits *= 0.5

    def context_block(self, *, max_chars: int = 2000) -> str:
        parts: list[str] = []
        if self.route is not None:
            parts.append(
                f"[STM route pathway={self.route.pathway_key} "
                f"D_eff={self.route.D_eff} domain={self.route.lean_domain} "
                f"emission={self.route.emission} recent_hits={self.recent_hits:.2f}]"
            )
        if self.active_ltm:
            parts.append("[STM←LTM retrieved]")
            for r in self.active_ltm[:4]:
                parts.append(
                    f"- (id={r.id} path={r.pathway} D={r.D_eff:.0f} str={r.strength:.2f}) "
                    f"{r.content[:280]}"
                )
        if self.turns:
            parts.append("[STM dialogue]")
            for t in list(self.turns)[-6:]:
                parts.append(f"{t.role}: {t.content[:200]}")
        text = "\n".join(parts)
        return text[:max_chars]


# ---------------------------------------------------------------------------
# Process layer — GPU/CPU forward only
# ---------------------------------------------------------------------------
@dataclass
class ProcessBinding:
    """What is allowed on the accelerator this step."""

    device: str  # cuda | cpu
    pathway: str
    max_new_tokens: int
    use_ltm: bool = True
    use_stm: bool = True


class FSOTMemoryOrganism:
    """
    Unified mind: LTM (disk) + STM (RAM) + process (GPU/CPU) + domain route.
    """

    def __init__(
        self,
        *,
        ltm_path: Optional[Path] = None,
        stm_capacity: int = 12,
        prefer_cuda: bool = True,
    ) -> None:
        self.ltm = LongTermMemory(ltm_path)
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.prefer_cuda = prefer_cuda
        self._pathway_cache: dict[str, Any] = {}

    def device(self) -> str:
        import torch

        if self.prefer_cuda and torch.cuda.is_available():
            try:
                major, _ = torch.cuda.get_device_capability(0)
                if major >= 7:
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def observe(self, user_text: str, pack_id: Optional[str] = None) -> DomainAllocation:
        alloc = resolve_allocation(user_text, pack_id=pack_id)
        self.stm.set_route(alloc)
        # Pull LTM matching this fold (disk → RAM only)
        hits = self.ltm.retrieve(
            query=user_text,
            pathway=alloc.pathway_key,
            top_k=4,
        )
        if not hits:
            hits = self.ltm.retrieve(query=user_text, top_k=3)
        self.stm.load_ltm_slice(hits)
        self.stm.push("user", user_text, pathway=alloc.pathway_key)
        return alloc

    def remember(
        self,
        content: str,
        *,
        pathway: Optional[str] = None,
        source: str = "session",
        tags: Optional[list[str]] = None,
    ) -> int:
        alloc = self.stm.route
        return self.ltm.store(
            content,
            pathway=pathway or (alloc.pathway_key if alloc else "ontology"),
            domain=alloc.lean_domain if alloc else "consciousness",
            D_eff=float(alloc.D_eff) if alloc else 16.0,
            emission=alloc.emission if alloc else "free",
            tags=tags or [],
            source=source,
        )

    def build_process_prompt(self, user_text: str) -> str:
        """STM block + user — what process sees (text only)."""
        block = self.stm.context_block()
        if block:
            return f"{block}\n\n[Process query]\n{user_text}"
        return user_text

    def process_generate(
        self,
        user_text: str,
        *,
        pack_id: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        remember_exchange: bool = True,
    ) -> dict[str, Any]:
        """
        Full cycle: observe → LTM retrieve → STM assemble → process on GPU/CPU
        → optional write-back to LTM.
        """
        from fsot_llm.domain_routing import inject_route_context
        from fsot_llm.pathway_adapters import load_pathway_observer

        alloc = self.observe(user_text, pack_id=pack_id)
        tok_budget = max_new_tokens or {
            "math": 512,
            "code": 384,
            "mcq": 64,
            "ontology": 128,
        }.get(alloc.pathway_key, 256)

        prompt = inject_route_context(
            self.build_process_prompt(user_text),
            alloc,
        )
        dev = self.device()
        binding = ProcessBinding(
            device=dev,
            pathway=alloc.pathway_key,
            max_new_tokens=tok_budget,
        )

        # Process: only active pathway adapter on accelerator
        obs = load_pathway_observer(alloc.pathway_key)
        text = obs.generate_text(
            prompt, max_new_tokens=tok_budget, temperature=0.0
        )
        self.stm.push("assistant", text, pathway=alloc.pathway_key)

        if remember_exchange:
            self.remember(
                f"Q: {user_text[:400]}\nA: {text[:600]}",
                source="process_writeback",
                tags=["dialogue", alloc.pathway_key],
            )

        return {
            "text": text,
            "allocation": {
                "pack_id": alloc.pack_id,
                "pathway": alloc.pathway_key,
                "D_eff": alloc.D_eff,
                "domain": alloc.lean_domain,
                "emission": alloc.emission,
            },
            "process": asdict(binding),
            "stm_ltm_hits": len(self.stm.active_ltm),
            "ltm_total": self.ltm.count(),
            "ltm_path": str(self.ltm.db_path),
        }

    def seed_from_topic_bank(self) -> int:
        """Import compact topic bank exemplars into LTM once."""
        bank_path = workspace_root() / "llm" / "data" / "memory" / "topic_bank.json"
        if not bank_path.is_file():
            return 0
        data = json.loads(bank_path.read_text(encoding="utf-8"))
        n = 0
        for tid, ent in (data.get("entries") or {}).items():
            topic = ent.get("topic") or {}
            for ex in ent.get("exemplars") or []:
                self.ltm.store(
                    ex,
                    pathway=(topic.get("tags") or ["ontology"])[0]
                    if topic.get("tags")
                    else "ontology",
                    domain="consciousness",
                    D_eff=float(topic.get("D_eff") or 16.0),
                    emission="free",
                    tags=list(topic.get("tags") or []) + [tid],
                    source="topic_bank",
                )
                n += 1
        return n


def main() -> int:
    ap = argparse.ArgumentParser(description="FSOT LTM/STM/process memory")
    ap.add_argument("--seed-bank", action="store_true", help="import topic_bank into LTM")
    ap.add_argument("--store", type=str, default=None, help="store a memory string")
    ap.add_argument("--query", type=str, default=None, help="retrieve LTM by query")
    ap.add_argument(
        "--chat",
        type=str,
        default=None,
        help="one-shot observe+process generate",
    )
    ap.add_argument("--pack", type=str, default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    org = FSOTMemoryOrganism()
    if args.seed_bank:
        n = org.seed_from_topic_bank()
        print(json.dumps({"seeded": n, "ltm": str(org.ltm.db_path), "count": org.ltm.count()}))
    if args.store:
        rid = org.remember(args.store, source="cli")
        print(json.dumps({"stored_id": rid, "count": org.ltm.count()}))
    if args.query is not None:
        hits = org.ltm.retrieve(query=args.query, top_k=5)
        print(
            json.dumps(
                [
                    {
                        "id": h.id,
                        "pathway": h.pathway,
                        "D_eff": h.D_eff,
                        "strength": h.strength,
                        "content": h.content[:200],
                    }
                    for h in hits
                ],
                indent=2,
            )
        )
    if args.chat:
        rep = org.process_generate(args.chat, pack_id=args.pack)
        print(json.dumps({k: rep[k] for k in rep if k != "text"}, indent=2))
        print("---")
        print(rep["text"][:800])
    if args.status or not any([args.seed_bank, args.store, args.query is not None, args.chat]):
        print(
            json.dumps(
                {
                    "ltm_path": str(org.ltm.db_path),
                    "ltm_count": org.ltm.count(),
                    "device": org.device(),
                    "stm_capacity": org.stm.capacity,
                    "layers": {
                        "LTM": "disk SQLite — long-term waveform",
                        "STM": "RAM — working set + retrieved LTM",
                        "process": "GPU/CPU — active pathway forward only",
                    },
                },
                indent=2,
            )
        )
    org.ltm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
