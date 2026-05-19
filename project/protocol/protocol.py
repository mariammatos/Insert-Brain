# ============================================================
# FILE: protocol/protocol.py
# ============================================================

import random

from psychopy import visual
from psychopy import core
from psychopy import event

from protocol.stimuli import Stimuli


class MotorImageryProtocol:

    def __init__(self, config, logger):

        self.config = config

        self.logger = logger

        self.clock = core.Clock()

        self.win = visual.Window(
            size=(1200, 800),
            fullscr=False,
            color="black",
            units="height"
        )

        self.stimuli = Stimuli(self.win)

        self.trials = self.generate_trials()

    # ========================================================
    # SAFETY EXIT
    # ========================================================

    def check_escape(self):

        keys = event.getKeys()

        if "escape" in keys:

            print("Experiment aborted.")

            self.win.close()

            core.quit()

    # ========================================================
    # TRIAL GENERATION
    # ========================================================

    def generate_trials(self):

        trials = []

        for class_name in self.config["classes"]:

            for _ in range(
                self.config["trials_per_class"]
            ):

                trials.append(class_name)

        random.shuffle(trials)

        return trials

    # ========================================================
    # BASELINE
    # ========================================================

    def baseline(self):

        # ==============================
        # EYES OPEN
        # ==============================

        self.stimuli.show_message(
            "BASELINE\n\n"
            "Eyes Open\n\n"
            "Relax and look at the center\n\n"
            "Press SPACE"
        )

        event.waitKeys(keyList=["space"])

        start = self.clock.getTime()

        self.logger.add_marker(
            start,
            "baseline_open_start",
            -1
        )

        self.stimuli.show_fixation()

        core.wait(
            self.config["baseline_open"]
        )

        end = self.clock.getTime()

        self.logger.add_marker(
            end,
            "baseline_open_end",
            -1
        )

        # ==============================
        # EYES CLOSED
        # ==============================

        self.stimuli.show_message(
            "BASELINE\n\n"
            "Eyes Closed\n\n"
            "Relax\n\n"
            "Press SPACE"
        )

        event.waitKeys(keyList=["space"])

        start = self.clock.getTime()

        self.logger.add_marker(
            start,
            "baseline_closed_start",
            -1
        )

        self.stimuli.show_fixation()

        core.wait(
            self.config["baseline_closed"]
        )

        end = self.clock.getTime()

        self.logger.add_marker(
            end,
            "baseline_closed_end",
            -1
        )

    # ========================================================
    # SINGLE TRIAL
    # ========================================================

    def run_trial(
        self,
        class_name,
        idx,
        total
    ):

        self.check_escape()

        class_info = self.config["classes"][class_name]

        label = class_info["label"]

        symbol = class_info["symbol"]

        # ==============================
        # FIXATION
        # ==============================

        self.stimuli.show_fixation()

        core.wait(
            self.config["fixation_time"]
        )

        # ==============================
        # CUE ON
        # ==============================

        cue_start = self.clock.getTime()

        self.logger.add_marker(
            cue_start,
            "cue_on",
            label
        )

        self.stimuli.show_cue(
            symbol,
            class_name
        )

        core.wait(
            self.config["cue_time"]
        )

        # ==============================
        # MOTOR IMAGERY START
        # ==============================

        mi_start = self.clock.getTime()

        self.logger.add_marker(
            mi_start,
            "mi_start",
            label
        )

        self.stimuli.show_cue(
            symbol,
            f"{class_name} ({idx+1}/{total})"
        )

        core.wait(
            self.config["imagery_time"]
        )

        # ==============================
        # MOTOR IMAGERY END
        # ==============================

        mi_end = self.clock.getTime()

        self.logger.add_marker(
            mi_end,
            "mi_end",
            label
        )

        # ==============================
        # REST
        # ==============================

        self.stimuli.show_fixation()

        core.wait(
            self.config["rest_time"]
        )

    # ========================================================
    # RUN FULL SESSION
    # ========================================================

    def run(self):

        self.baseline()

        total = len(self.trials)

        for idx, class_name in enumerate(self.trials):

            self.run_trial(
                class_name,
                idx,
                total
            )

        self.stimuli.show_message(
            "Session Complete!\n\n"
            "Thank you."
        )

        core.wait(5)

        self.win.close()