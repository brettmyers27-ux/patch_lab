from core.serum2_state_reconstruct import (
    HostStateTemplate,
    XferState,
    decode_host_template,
    encode_host_template,
)


def test_complete_host_template_round_trips_both_chunks() -> None:
    template = HostStateTemplate(
        class_id="56535453657232736572756D20320000",
        component=XferState({"kind": "component"}, 2, {"Oscillator0": {"value": 1.0}}),
        controller=XferState({"kind": "controller"}, 2, {"ModSlot0": {"amount": 32.0}}),
    )

    decoded = decode_host_template(encode_host_template(template))

    assert decoded.class_id == template.class_id
    assert decoded.component.data == template.component.data
    assert decoded.controller.data == template.controller.data
