"""Authoritative compatibility contract for permanent personal-preset Match data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from core.runtime_compatibility import CLAP_CHECKPOINT_SHA256


# These are data-format revisions, deliberately independent of the PatchLab app
# version. Changing one invalidates only the permanent data it describes.
RENDER_REVISION = "seven-notes-44k1-float-v1"
FINGERPRINT_REVISION = "clap512-handcrafted9-per-note-mean-v1"
SERUM2_FINGERPRINT_REVISION = "clap512-handcrafted9-audible-notes-mean-v2"
CLAP_REVISION = f"sha256:{CLAP_CHECKPOINT_SHA256}"
HANDCRAFTED_REVISION = "spectral-audio-features-9-v1"
SERUM1_SCHEMA_REVISION = "serum1-normalized-parameters-v1"
SERUM2_SCHEMA_REVISION = "serum2-full-decoded-settings-v1"

RENDER_MIDI_NOTES = (24, 36, 48, 60, 72, 84, 96)
REQUIRED_FINGERPRINT_NOTES = (0, *RENDER_MIDI_NOTES)
CLAP_FLOAT_COUNT = 512
HANDCRAFTED_FLOAT_COUNT = 9
FLOAT32_BYTES = 4


def audible_notes_from_mask(mask: int | None) -> tuple[int, ...]:
    """Return actual covered notes, or the legacy full range when absent."""

    if mask is None:
        return RENDER_MIDI_NOTES
    value = int(mask)
    return tuple(
        note for index, note in enumerate(RENDER_MIDI_NOTES)
        if value & (1 << index)
    )


def nearest_audible_note(requested: int, mask: int | None) -> int:
    """Choose a playable note while retaining the original requested octave."""

    notes = audible_notes_from_mask(mask)
    if not notes:
        raise ValueError("no audible MIDI notes were recorded")
    return min(notes, key=lambda note: (abs(note - requested), note))


@dataclass(frozen=True, slots=True)
class PreparedRevision:
    render: str = RENDER_REVISION
    fingerprint: str = FINGERPRINT_REVISION
    serum2_fingerprint: str = SERUM2_FINGERPRINT_REVISION
    clap: str = CLAP_REVISION
    handcrafted: str = HANDCRAFTED_REVISION
    serum1: str = SERUM1_SCHEMA_REVISION
    serum2: str = SERUM2_SCHEMA_REVISION

    def fingerprint_for(self, synth: str) -> str:
        return self.serum2_fingerprint if synth == "serum2" else self.fingerprint

    def sql_parameters(self) -> dict[str, object]:
        return {
            "prepared_render_revision": self.render,
            "prepared_fingerprint_revision": self.fingerprint,
            "prepared_serum2_fingerprint_revision": self.serum2_fingerprint,
            "prepared_clap_revision": self.clap,
            "prepared_handcrafted_revision": self.handcrafted,
            "prepared_serum1_revision": self.serum1,
            "prepared_serum2_revision": self.serum2,
            "prepared_note_count": len(REQUIRED_FINGERPRINT_NOTES),
            "prepared_embedding_bytes": CLAP_FLOAT_COUNT * FLOAT32_BYTES,
            "prepared_handcrafted_bytes": HANDCRAFTED_FLOAT_COUNT * FLOAT32_BYTES,
        }


CURRENT_PREPARED_REVISION = PreparedRevision()


def prepared_revision_token(
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> str:
    """Stable job token proving which data contract an interrupted run targeted."""

    payload = {
        "clap": revision.clap,
        "fingerprint": revision.fingerprint,
        "serum2_fingerprint": revision.serum2_fingerprint,
        "handcrafted": revision.handcrafted,
        "render": revision.render,
        "serum1": revision.serum1,
        "serum2": revision.serum2,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepared_predicate(
    preset_alias: str = "p",
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
    require_active_source: bool = True,
    require_revision: bool = True,
) -> tuple[str, dict[str, object]]:
    """Return the single SQL definition of a prepared preset.

    The caller embeds the predicate in its query. Match, the work queue, and
    the public validation API all use this exact contract.
    """

    alias = preset_alias
    clauses: list[str] = []
    if require_active_source:
        clauses.append(
            f"EXISTS (SELECT 1 FROM preset_sources ps "
            f"WHERE ps.preset_id={alias}.id AND ps.active=1 "
            f"AND ps.content_hash={alias}.content_hash)"
        )
    if require_revision:
        clauses.append(
            "EXISTS (SELECT 1 FROM prepared_presets pr "
            f"WHERE pr.preset_id={alias}.id "
            "AND pr.render_revision=:prepared_render_revision "
            "AND pr.clap_revision=:prepared_clap_revision "
            "AND pr.handcrafted_revision=:prepared_handcrafted_revision "
            "AND (("
            f"{alias}.synth='serum1' "
            "AND pr.fingerprint_revision=:prepared_fingerprint_revision "
            "AND pr.serum1_schema_revision=:prepared_serum1_revision"
            ") OR ("
            f"{alias}.synth='serum2' "
            "AND pr.fingerprint_revision=:prepared_serum2_fingerprint_revision "
            "AND pr.serum2_schema_revision=:prepared_serum2_revision"
            ")))"
        )
    all_notes = ",".join(str(note) for note in REQUIRED_FINGERPRINT_NOTES)
    render_notes = ",".join(str(note) for note in RENDER_MIDI_NOTES)
    valid_blob = (
        "length(pf.embedding_f32)=:prepared_embedding_bytes "
        "AND length(pf.handcrafted_f32)=:prepared_handcrafted_bytes"
    )
    full = (
        "(SELECT COUNT(*) FROM fingerprints pf "
        f"WHERE pf.preset_id={alias}.id AND pf.midi_note IN ({all_notes}) "
        f"AND {valid_blob})=:prepared_note_count"
    )
    aggregate = (
        "(SELECT COUNT(*) FROM fingerprints pf "
        f"WHERE pf.preset_id={alias}.id AND pf.midi_note=0 "
        f"AND {valid_blob})=1"
    )
    note_bits = " ".join(
        f"WHEN {note} THEN {1 << index}"
        for index, note in enumerate(RENDER_MIDI_NOTES)
    )
    coverage = (
        "EXISTS (SELECT 1 FROM fingerprint_note_coverage nc "
        f"WHERE nc.preset_id={alias}.id AND nc.note_mask BETWEEN 1 AND 127 "
        "AND nc.note_mask=COALESCE((SELECT SUM(CASE pf.midi_note "
        f"{note_bits} ELSE 0 END) FROM fingerprints pf "
        f"WHERE pf.preset_id={alias}.id AND pf.midi_note IN ({render_notes}) "
        f"AND {valid_blob}),0) "
        "AND NOT EXISTS (SELECT 1 FROM fingerprints pf "
        f"WHERE pf.preset_id={alias}.id AND pf.midi_note IN ({render_notes}) "
        f"AND NOT ({valid_blob})))"
    )
    clauses.append(
        f"(({alias}.synth='serum1' AND {full}) OR "
        f"({alias}.synth='serum2' AND {aggregate} "
        f"AND {coverage}))"
    )
    clauses.append(
        "(("
        f"{alias}.synth='serum1' AND EXISTS (SELECT 1 FROM params pa "
        f"WHERE pa.preset_id={alias}.id)"
        ") OR ("
        f"{alias}.synth='serum2' AND EXISTS (SELECT 1 FROM serum2_full_settings s2 "
        f"WHERE s2.preset_id={alias}.id)"
        "))"
    )
    return " AND ".join(f"({clause})" for clause in clauses), revision.sql_parameters()


def is_preset_prepared(
    connection: sqlite3.Connection,
    preset_id: int,
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> bool:
    """Validate that one preset is active, complete, and revision compatible."""

    predicate, parameters = prepared_predicate(revision=revision)
    parameters = {**parameters, "prepared_preset_id": int(preset_id)}
    row = connection.execute(
        f"SELECT EXISTS(SELECT 1 FROM presets p "
        f"WHERE p.id=:prepared_preset_id AND {predicate})",
        parameters,
    ).fetchone()
    return bool(row[0])


def has_required_permanent_data(
    connection: sqlite3.Connection,
    preset_id: int,
) -> bool:
    """Validate feature and Serum state before recording a prepared revision."""

    predicate, parameters = prepared_predicate(
        require_active_source=False,
        require_revision=False,
    )
    parameters = {**parameters, "prepared_preset_id": int(preset_id)}
    row = connection.execute(
        f"SELECT EXISTS(SELECT 1 FROM presets p "
        f"WHERE p.id=:prepared_preset_id AND {predicate})",
        parameters,
    ).fetchone()
    return bool(row[0])


def record_prepared_revision(
    connection: sqlite3.Connection,
    preset_id: int,
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> bool:
    """Record the current revisions only after validating permanent data."""

    if not has_required_permanent_data(connection, preset_id):
        return False
    synth_row = connection.execute(
        "SELECT synth FROM presets WHERE id=?", (preset_id,)
    ).fetchone()
    if synth_row is None:
        return False
    connection.execute(
        """
        INSERT INTO prepared_presets(
          preset_id,render_revision,fingerprint_revision,clap_revision,
          handcrafted_revision,serum1_schema_revision,serum2_schema_revision
        ) VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(preset_id) DO UPDATE SET
          render_revision=excluded.render_revision,
          fingerprint_revision=excluded.fingerprint_revision,
          clap_revision=excluded.clap_revision,
          handcrafted_revision=excluded.handcrafted_revision,
          serum1_schema_revision=excluded.serum1_schema_revision,
          serum2_schema_revision=excluded.serum2_schema_revision,
          prepared_at=CURRENT_TIMESTAMP
        """,
        (
            int(preset_id),
            revision.render,
            revision.fingerprint_for(str(synth_row[0])),
            revision.clap,
            revision.handcrafted,
            revision.serum1,
            revision.serum2,
        ),
    )
    return True
