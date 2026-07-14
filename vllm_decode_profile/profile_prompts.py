"""Self-contained meaningful long-prompt construction."""

from __future__ import annotations


DEEPSEEK_USER_TOKEN = "<｜User｜>"
DEEPSEEK_ASSISTANT_TOKEN = "<｜Assistant｜>"


def _prompt_wrapper_ids(tokenizer) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(DEEPSEEK_USER_TOKEN, add_special_tokens=False)
    suffix_ids = tokenizer.encode(
        DEEPSEEK_ASSISTANT_TOKEN,
        add_special_tokens=False,
    )
    if tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids
    return prefix_ids, suffix_ids


def _meaningful_long_qa_text() -> tuple[str, str, str]:
    intro = """
Read the following long archive packet and answer the final question with one
short phrase.

Archive packet title: The Hawthorn Bridge Water Project.

The packet describes a valley repair project that rebuilt a temporary supply
crossing after a flood. The important distinction is that Willow Bridge was an
early footbridge that washed out, while Hawthorn Bridge was the later temporary
bridge used by supply carts during the signed final inspection.
"""
    evidence_block = """
Ledger section: The Silver Orchard water project began as a repair plan for
three villages. Mira Patel kept the repair ledger. Elena Ruiz supervised the
engineering notes. Early in the project, workers built Willow Bridge from pine
boards and rope. Willow Bridge was intended only for foot traffic. After three
days of heavy rain, Willow Bridge washed out and was marked with a red circle
on the hazard map.

Replacement section: The committee then built a second temporary bridge with
iron pins, ash beams, and a gravel approach. The second bridge was named
Hawthorn Bridge because hawthorn trees marked the crossing. After the flood,
every delivery receipt used the name Hawthorn Bridge. The bridge carried lime,
pump parts, sacks of oats, and the spare intake screen. Loaded carts crossed
one at a time. The archive index says Hawthorn Bridge was the active temporary
supply crossing during the final inspection.

Inspection section: The final inspection was held in the schoolhouse. The
committee asked whether the channel leaked, whether the pump could run for two
hours, and whether the temporary bridge was safe for supply carts. The signed
inspection notes say the temporary bridge was safe for carts only if carts
crossed one at a time. The bridge name in those signed inspection notes was
Hawthorn Bridge, not Willow Bridge. Mira Patel signed the notes, and Elena Ruiz
countersigned them.

Correction section: A later clerk wrote a confusing margin note mentioning
Willow Bridge beside the final inspection. The staff correction says the margin
note was copied from the first week of repairs and should not override the
signed inspection notes. The typed archive copy preserves the signed inspection
wording and removes the mistaken margin note. The search term Willow Bridge
points to the flood damage file. The search term Hawthorn Bridge points to the
final inspection file.

Audit section: HAWTHORN-ACTIVE-CROSSING, HAWTHORN-SUPPLY-MAP,
HAWTHORN-CART-ROUTE, and HAWTHORN-FINAL-NOTE all refer to the same temporary
bridge. WILLOW-DAMAGE-FILE, WILLOW-FLOOD-NOTE, and WILLOW-OLD-DRAFT refer to
the washed-out early bridge. The answer should use the bridge name from the
signed final inspection notes.
"""
    final_section = """
Final evidence summary:
1. Willow Bridge was the early footbridge and washed out before the final
inspection.
2. Hawthorn Bridge was the later temporary supply crossing.
3. The signed final inspection notes identify the temporary bridge as
Hawthorn Bridge.
4. The later Willow Bridge margin note is explicitly marked as a mistake.

Question: What was the name of the temporary bridge used during the final
inspection?

Answer with exactly the bridge name.
"""
    return intro, evidence_block, final_section


def build_base_meaningful_prompt(
    tokenizer,
    base_target_len: int = 10000,
) -> list[int]:
    if base_target_len <= 0:
        raise ValueError("base_target_len must be positive")

    prefix_ids, suffix_ids = _prompt_wrapper_ids(tokenizer)
    intro, evidence_block, final_section = _meaningful_long_qa_text()
    intro_ids = tokenizer.encode(intro, add_special_tokens=False)
    evidence_ids = tokenizer.encode(evidence_block, add_special_tokens=False)
    final_ids = tokenizer.encode(final_section, add_special_tokens=False)

    fixed_len = len(prefix_ids) + len(intro_ids) + len(final_ids) + len(suffix_ids)
    if base_target_len <= fixed_len:
        body_ids = intro_ids + final_ids
        return (prefix_ids + body_ids + suffix_ids)[-base_target_len:]

    evidence_budget = base_target_len - fixed_len
    repeats = (evidence_budget + len(evidence_ids) - 1) // len(evidence_ids)
    evidence = (evidence_ids * repeats)[:evidence_budget]
    return prefix_ids + intro_ids + evidence + final_ids + suffix_ids


def build_exact_token_prompt(
    base_ids: list[int],
    target_len: int,
) -> list[int]:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if not base_ids:
        raise ValueError("base prompt must not be empty")
    repeats = (target_len + len(base_ids) - 1) // len(base_ids)
    return (base_ids * repeats)[-target_len:]


def decode_prompt_tail(
    tokenizer,
    token_ids: list[int],
    max_chars: int = 300,
) -> str:
    try:
        text = tokenizer.decode(token_ids[-512:], skip_special_tokens=False)
    except TypeError:
        text = tokenizer.decode(token_ids[-512:])
    text = " ".join(text.split())
    return text[-max_chars:]
