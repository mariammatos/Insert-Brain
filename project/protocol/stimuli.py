# ============================================================
# FILE: protocol/stimuli.py
#
# Manages all visual stimuli for the experiment window.
# Handles fixation cross, cue display, HUD elements,
# progress bar, and phase labels using PsychoPy.
# ============================================================

from psychopy import visual


class Stimuli:

    def __init__(self, win):

        self.win = win

        # ───────────── Main stimuli ──────────────────────

        self.fixation = visual.TextStim(
            win,
            text="+",
            height=0.15,
            color="white"
        )

        self.cue = visual.TextStim(
            win,
            text="",
            height=0.25,
            color="white"
        )

        self.info = visual.TextStim(
            win,
            text="",
            pos=(0, -0.3),
            height=0.05,
            color="white"
        )

        # # ── HUD — top left corner ────────────────────────
        # Remaining time
        self.hud_time = visual.TextStim(
            win,
            text="",
            pos=(-0.75, 0.45),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="left"
        )

        # Completed trials
        self.hud_trials = visual.TextStim(
            win,
            text="",
            pos=(-0.75, 0.39),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="left"
        )

        # ── HUD — top right corner ───────────────────────
        # Current class
        self.hud_class = visual.TextStim(
            win,
            text="",
            pos=(0.75, 0.45),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="right"
        )

        # ── Time progress bar ────────────────────────────
        # Grey background
        self.bar_bg = visual.Rect(
            win,
            width=1.4,
            height=0.025,
            pos=(0, -0.46),
            fillColor="#333333",
            lineColor=None
        )

        # Green fill (shrinks over time)
        self.bar_fill = visual.Rect(
            win,
            width=1.4,
            height=0.025,
            pos=(0, -0.46),
            fillColor="#44BB77",
            lineColor=None
        )

        # ── Trial phase label (FIXATION / CUE / IMAGERY / REST) ─
        self.phase_text = visual.TextStim(
            win,
            text="",
            pos=(0, 0.38),
            height=0.04,
            color="#888888"
        )

        # Stores total time for progress bar calculation
        self._max_seconds = 1.0

    # ========================================================
    # SET MAX TIME (call at the start of the protocol)
    # ========================================================

    def set_max_time(self, max_seconds):
        self._max_seconds = max_seconds

    # ========================================================
    # UPDATE HUD
    # Call before any draw, passes current values
    # ========================================================

    def update_hud(self, elapsed_seconds, trials_done, current_class=""):
        """
        Update HUD elements without drawing them yet.
        elapsed_seconds : seconds since the start of the session
        trials_done     : number of completed trials
        current_class   : current class name (e.g. "LEFT")
        """

        remaining = max(0.0, self._max_seconds - elapsed_seconds)
        mins      = int(remaining // 60)
        secs      = int(remaining % 60)

        self.hud_time.text   = f"⏱  {mins}:{secs:02d}"
        self.hud_trials.text = f"✓  {trials_done} trials"
        self.hud_class.text  = current_class

        # Progress bar — shrinks from right to left
        ratio     = max(0.0, remaining / self._max_seconds)
        bar_width = 1.4 * ratio
        self.bar_fill.width = max(0.001, bar_width)
        self.bar_fill.pos   = (-0.7 + bar_width / 2, -0.46)

        # Colour changes to orange in last 2 min, red in last 1 min
        if remaining < 60:
            self.bar_fill.fillColor = "#DD4444"
        elif remaining < 120:
            self.bar_fill.fillColor = "#DDAA33"
        else:
            self.bar_fill.fillColor = "#44BB77"

    def _draw_hud(self):
        """Draw all HUD elements."""
        self.bar_bg.draw()
        self.bar_fill.draw()
        self.hud_time.draw()
        self.hud_trials.draw()
        if self.hud_class.text:
            self.hud_class.draw()

    # ========================================================
    # DRAW TRIAL PHASE
    # ========================================================

    def _draw_phase(self, phase):
        """Draw the current phase text (FIXATION, IMAGERY, etc.)."""
        self.phase_text.text = phase
        self.phase_text.draw()

    # ========================================================
    # PUBLIC METHODS — compatible with original + HUD
    # ========================================================

    def show_fixation(self, elapsed=None, trials_done=None):
        """
        Fixation cross.
        If elapsed and trials_done are passed, show the HUD.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0)
            self._draw_hud()

        self._draw_phase("FIXATION")
        self.fixation.draw()
        self.win.flip()

    def show_cue(self, symbol, text, elapsed=None, trials_done=None):
        """
        Cue arrow/symbol + class text.
        If elapsed and trials_done are passed, show the HUD.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0, current_class=text)
            self._draw_hud()

        self._draw_phase("IMAGERY")
        self.cue.text  = symbol
        self.info.text = text
        self.cue.draw()
        self.info.draw()
        self.win.flip()

    def show_imagery(self, symbol, text, elapsed=None, trials_done=None):
        """
        Same as show_cue but with the IMAGERY phase highlighted.
        Uses a brighter color to indicate the active window.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0, current_class=text)
            self._draw_hud()

        self._draw_phase("★  IMAGERY  ★")
        self.cue.text  = symbol
        self.info.text = text

        # Highlight the symbol during the imagery interval
        original_color = self.cue.color
        self.cue.color = "#FFDD44"
        self.cue.draw()
        self.cue.color = original_color

        self.info.draw()
        self.win.flip()

    def show_rest(self, elapsed=None, trials_done=None):
        """Rest cross between trials with HUD."""

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0)
            self._draw_hud()

        self._draw_phase("REST")
        self.fixation.draw()
        self.win.flip()

    def show_message(self, message):
        """Full-screen message (same as original, without HUD)."""

        msg = visual.TextStim(
            self.win,
            text=message,
            height=0.06,
            wrapWidth=1.5,
            color="white"
        )

        msg.draw()
        self.win.flip()

    def show_trial_feedback(self, trial_num, class_name, symbol,
                             elapsed, trials_done):
        """
        Brief feedback between trials:
        shows which class the finished trial was.
        """

        self.update_hud(elapsed, trials_done, current_class="")
        self._draw_hud()

        msg = visual.TextStim(
            self.win,
            text=f"{symbol}\n{class_name}",
            height=0.08,
            color="#888888"
        )
        msg.draw()
        self.win.flip()
