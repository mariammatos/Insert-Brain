# ============================================================
# FILE: protocol/protocol.py
# ============================================================

import time
import random

from psychopy import visual
from psychopy import core
from psychopy import event

from protocol.stimuli import Stimuli


class MotorImageryProtocol:

    def __init__(self, config, logger):

        self.config = config

        self.logger = logger

        # Guardar o offset Unix no exacto momento em que o Clock é criado.
        # Permite converter qualquer self.clock.getTime() para Unix time:
        #   unix_ts = self.clock_unix_offset + self.clock.getTime()
        #
        # É crítico que as duas linhas abaixo sejam consecutivas e sem
        # nenhum código entre elas, para minimizar o erro de medição.
        self.clock             = core.Clock()
        self.clock_unix_offset = time.time()

        self.win = visual.Window(
            size=(1200, 800),
            fullscr=False,
            color="black",
            units="height"
        )

        self.stimuli = Stimuli(self.win)

        self.trials = self.generate_trials()

    # ========================================================
    # HELPER: converter tempo do Clock para Unix time
    # ========================================================

    def to_unix(self, clock_time):
        """
        Converte um timestamp de self.clock.getTime() para Unix time,
        usando o offset calculado no __init__.

        O pequeno erro entre as duas chamadas (~microsegundos) é
        desprezável para EEG a 250 Hz (resolução de 4 ms).
        """
        return self.clock_unix_offset + clock_time

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
            self.to_unix(start),
            "baseline_open_start",
            -1
        )

        self.stimuli.show_fixation()

        core.wait(
            self.config["baseline_open"]
        )

        end = self.clock.getTime()

        self.logger.add_marker(
            self.to_unix(end),
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
            self.to_unix(start),
            "baseline_closed_start",
            -1
        )

        self.stimuli.show_fixation()

        core.wait(
            self.config["baseline_closed"]
        )

        end = self.clock.getTime()

        self.logger.add_marker(
            self.to_unix(end),
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
            self.to_unix(cue_start),
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
            self.to_unix(mi_start),
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
            self.to_unix(mi_end),
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