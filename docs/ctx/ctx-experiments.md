# `ctx` Prompt Enhancement Experiments

A running log of default vs unicorn (`enhance_prompt` vs `enhance_unicorn`) behavior while tuning `ctx.py` for Pirates Online Classic.

---

## Configuration Timeline

| Date | Change | Notes |
|------|--------|-------|
| Earlier | Baseline | `rewrite_max_tokens` hard-coded 320, unicorn Pass 2 capped at 240, hook timeout 30s. |
| Nov 21 2025 | `rewrite_max_tokens` exposed via `ctx_config.json` / hook env | Enables CLI/Web hook to set `CTX_REWRITE_MAX_TOKENS`; config now 420. |
| Nov 21 2025 | Unicorn Pass 2 budget | Pass 2 `max_tokens=min(300, budget)`; Pass 1 remains `min(180, budget)`. |
| Nov 21 2025 | Hook timeout | `timeout 30s` → `timeout 60s` to avoid unicorn fallbacks on slow MCP/decoder responses. |

Current relevant settings (`ctx_config.json`):

```json
{
  "rewrite_max_tokens": 420,
  "default_mode": "unicorn",
  "require_context": true,
  "relevance_gate_enabled": false,
  "min_relevance": 0.05
}
```

Hook exports: `CTX_REWRITE_MAX_TOKENS=420`, `CTX_REQUIRE_CONTEXT=true`, `CTX_RELEVANCE_GATE=false`, `CTX_MIN_RELEVANCE=0.05`.

---

## Canonical Queries

### 1. Cannon Reload / Cooldown

- **Prompt**
  > Explain how cannon reload and cooldown timers are enforced for player ships...

- **Default mode**
  - References: `DistributedPCCannon.py` (264–273), `ShipGlobals.py` (7757–7816), `LocalPirate.py` (1801–1860), `ShipManagerAI.py` (≈50), `PirateAvatarPanel.py` (≈95).
  - Structure: two detailed paragraphs (dedup removes duplicates).

- **Unicorn mode**
  - Same core files plus broader context (quest modifiers, upgrades).
  - Emphasizes interactions (cooldowns ↔ quests/upgrades ↔ HUD sync).

### 2. Quest Progression

- **Prompt**
  > Describe how the quest progression system... quest states represented... objectives registered/updated... validation before advancing...

- **Default mode**
  - References: `QuestManagerAI`, `DistributedQuestAvatar`, `DistributedQuestGiver`, `QuestStepIndicator`, `DistributedBuriedTreasureAI`.
  - Broad overview of the involved components.

- **Unicorn mode**
  - Focus shifts to relationships: `QuestStepIndicator` ↔ `QuestManagerAI` ↔ `DistributedBuriedTreasureAI`/`DistributedSearchableContainerAI`.
  - Deeper look at step validation conditions.

### 3. Ship Repair at Sea

- **Prompt**
  > Describe how ship repair at sea works end-to-end... materials, cooldowns, crew/HUD coordination...

- **Default mode (420 tokens)**
  - References: `DistributedShopKeeper.py` (353–360), `ShipRepairSpotMgrBase.py` (4–51), `CombatTray.py` (617–619), `OTPInternalRepository.py` (45–59), `PLocalizerEnglish.py` (status strings).
  - Clean end-to-end breakdown.

- **Unicorn mode**
  - After Pass 2 bump + 60s timeout: rich, non-truncated output.
  - Details data structures, crew modifiers, state tracking, HUD updates, localized messaging.

### 4. Crew & Guild Invites / Membership

- **Prompt**
  > Describe how crew and guild invite and membership workflows are implemented...

- **Default mode**
  - References: `DistributedCrewMatchAI`, `GuiManager.py`, `CrewPage.py`, `GuildPage.py`.
  - High-level overview of UI + server components.

- **Unicorn mode**
  - With extended timeout: consistent rewrite.
  - Focus on GUI methods (GuiManager: 896–901, 917–922; GuildPage: 271–272, 338–341) and invite persistence/expiration flows.

### 5. Quest Pipeline Overview

- **Prompt**
  > Give a high-level overview of the quest pipeline in Pirates Online Classic, from NPC conversations to quest completion rewards, and where key state transitions are handled in the codebase.

- **Default mode**
  - References: `DistributedQuestGiver` (≈453–512), `QuestLadderDB`, `QuestTaskDNA`, `QuestDB`.
  - Good end-to-end story: NPC conversation → quest assignment → task tracking → rewards/persistence.

- **Unicorn mode**
  - Emphasizes the **sequence** of events:
    - NPC accept flow: `DistributedQuestGiver` methods + `QuestLadderDB` availability checks.
    - Task completion flow: `QuestManagerAI` (63–84) + `DistributedQuestGiver` (111–160) + `QuestLadderDB` ladder updates.
  - More explicitly pipeline-focused (who calls what, when state flips).

### 6. PVP / Faction System

- **Prompt**
  > Describe how the PVP and faction system is implemented, including how kills and deaths are tracked, how rank and reputation progression works, and where HUD updates and matchmaking are handled.

- **Default mode**
  - References: `DistributedPVPInstance.py` (251–310), `BattleManagerBase.py` (15–19), `PVPInviter.py` (284–289), `PVPRankGui.py` (128–141), `CombatTray.py` (1473–1476).
  - Clear mapping of responsibilities (kill/death tracking, battle state, matchmaking, rank GUI, HUD updates).

- **Unicorn mode**
  - Focuses on flows:
    - Kill/death events → stats structures → propagation to other systems and HUD.
    - Accumulated stats → rank calculations in `PVPRankGui` → GUI update triggers when rank changes.
  - Less enumeration, more description of how state moves between components.

### 7. Named Boss Loot & Treasure Drops

- **Prompt**
  > Describe how named boss loot tables and treasure drops are implemented, including how rarity weights are configured for different item tiers, how drop events are validated and logged, and where any related balancing or analytics hooks live.

- **Default mode**
  - References: `QuestTaskDNA.py` (2770–2798), `QuestStatus.py` (328–346), `MessageStackPanel.py` (474–479), `QuestBase.py` (11–12).
  - Good coverage of loot probability data structures, weighted random selection, quest/quest-status validation, and UI logging.

- **Unicorn mode**
  - Adds `LootPopupPanel.py` (101–160) to describe loot popup behavior.
  - More detail on:
    - Probability distributions for item tiers.
    - Runtime modification of loot tables and consistency checks across bosses.
    - Validation to prevent duplicates/invalid items and how drop events are logged for analytics.

### 8. Simple Prompt Sanity Check

- **Prompt 1**
  > Where is cannon reload and cooldown logic implemented in Pirates Online Classic?

  - **Default mode**
    - Points directly to a small set of functions/files (e.g. `CombatAnimations.finishReload`, `WeaponBase.__cannonPortHit`, `CutsceneActor.fireCannon`, tutorial cannon, `Cannon.__init__`).
    - Two concise paragraphs that already answer the “where is it implemented?” question.

  - **Unicorn mode**
    - Adds more intra-file detail (extra parameters, animation relationships, cutscene transitions).
    - Extra information but not clearly more useful for this simple lookup.

- **Prompt 2**
  > Which classes and files handle crew and guild invite workflows in Pirates Online Classic?

  - **Default mode**
    - References: `CrewInvitee.py`, `GuildInvitee.py`, `CrewMatchInvitee.py`, `GuiManager.py`.
    - Clean answer to “which classes/files are involved?”

  - **Unicorn mode**
    - Much deeper: inheritance (`RequestButton`, `GuiPanel.GuiPanel`), exact constructor signatures and ignored-player checks, partial description of `__handleCancelFromAbove` and `destroy`.
    - Overkill when the goal is just to locate the key classes.

---

## Observations

- **Default mode strengths**
  - Fast, reliable single-pass roadmaps.
  - Clear file/line anchors.

- **Unicorn mode strengths**
  - Better at surfacing cross-system interactions.
  - Two-pass retrieval (snippets → paths) keeps references concrete.
  - Extra Pass 2 headroom (300 tokens) reduces truncation for deep-dive prompts.
  - Requires longer timeout (60s) to avoid fallback due to multi-pass latency.

- **Fallback behavior**
  - With `require_context=true`, both modes return the original prompt when no usable context is found (or when MCP/decoder fails). This is expected and desired.

---

## Recommended Future Experiments

We have now exercised both **complex pipelines** (quests, repair, PVP, loot, crew/guild) and **simple lookups** (where/which classes). Future ideas if we want to push further:

1. **Non-Pirates repos**: run the same default vs unicorn comparisons on a different codebase to ensure behavior generalizes.
2. **Docs/memory-heavy queries**: questions that hit `context_search` with blended memories instead of primarily code.
3. **Detail mode vs unicorn**: compare `--detail` alone vs `--unicorn` on the same prompts to clarify when snippets are “enough” without multi-pass.

For any new experiment, record:

- File/line coverage
- Emphasis on relationships vs enumeration
- Whether unicorn’s extra passes + token budget add clear value over default.

---

_Last updated: Nov 21, 2025_
