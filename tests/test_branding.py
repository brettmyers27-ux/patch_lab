from core.branding import (
    display_match_name,
    generated_preset_name,
    public_match_name,
)


def test_closest_match_preserves_real_preset_name() -> None:
    assert display_match_name("BA - Lazer Bass", 1) == "BA - Lazer Bass"
    assert display_match_name("Lead & Pluck", 2) == "Lead & Pluck"


def test_closest_match_uses_branded_fallback_only_when_name_is_missing() -> None:
    assert display_match_name("", 3) == public_match_name(3)
    assert display_match_name(None, 4) == "PatchLab Library Match 4"


def test_legacy_masked_match_recovers_name_from_saved_source_path() -> None:
    assert (
        display_match_name(
            "PatchLab Library Match 1",
            1,
            source_path="Factory/Bass/Hard/BA - Lazer Bass.SerumPreset",
        )
        == "BA - Lazer Bass"
    )


def test_generated_presets_remain_patchlab_branded() -> None:
    assert generated_preset_name("serum1") == "PatchLab Serum 1 Match"
    assert generated_preset_name("serum2") == "PatchLab Serum 2 Match"
