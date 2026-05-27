# ============================================================
# FILE: protocol/protocol_timed.py
# ============================================================

import time
import random

from psychopy import visual, core, event

from protocol.stimuli import Stimuli

MAX_TRAIN_SECONDS = 9 * 60   # muda aqui se quiseres


class MotorImageryProtocolTimed:

    def __init__(self, config, logger):

        self.config  = config
        self.logger  = logger
        self.clock   = core.Clock()

        self.win = visual.Window(
            size=(1200, 800),
            fullscr=False,
            color="black",
            units="height"
        )

        self.stimuli = Stimuli(self.win)
        self.stimuli.set_max_time(MAX_TRAIN_SECONDS)

        self.class_names = list(self.config["classes"].keys())
        self._rr_index   = 0
        self._rr_order   = self.class_names.copy()
        random.shuffle(self._rr_order)

    # ========================================================
    # SAFETY EXIT
    # ========================================================

    def check_escape(self):
        if "escape" in event.getKeys():
            print("Experiment aborted.")
            self.win.close()
            core.quit()

    # ========================================================
    # ROUND-ROBIN
    # ========================================================

    def next_class(self):
        class_name      = self._rr_order[self._rr_index]
        self._rr_index += 1
        if self._rr_index >= len(self._rr_order):
            self._rr_index = 0
            random.shuffle(self._rr_order)
        return class_name

    # ========================================================
    # BASELINE
    # ========================================================

    def baseline(self):

        self.stimuli.show_message(
            "BASELINE\n\n"
            "Eyes Open\n\n"
            "Relax and look at the center\n\n"
            "Press SPACE"
        )
        event.waitKeys(keyList=["space"])

        t = time.time()
        self.logger.add_marker(t, "baseline_open_start", -1)
        self.stimuli.show_fixation()
        core.wait(self.config["baseline_open"])
        self.logger.add_marker(time.time(), "baseline_open_end", -1)

        self.stimuli.show_message(
            "BASELINE\n\n"
            "Eyes Closed\n\n"
            "Relax\n\n"
            "Press SPACE"
        )
        event.waitKeys(keyList=["space"])

        t = time.time()
        self.logger.add_marker(t, "baseline_closed_start", -1)
        self.stimuli.show_fixation()
        core.wait(self.config["baseline_closed"])
        self.logger.add_marker(time.time(), "baseline_closed_end", -1)

    # ========================================================
    # SINGLE TRIAL
    # ========================================================

    def run_trial(self, class_name, trials_done, session_start):
        """
        Corre um trial completo com HUD visível em todas as fases.
        Devolve False se o utilizador carregou SPACE para parar.
        """

        self.check_escape()

        class_info = self.config["classes"][class_name]
        label      = class_info["label"]
        symbol     = class_info["symbol"]

        def elapsed():
            return time.time() - session_start

        # ── FIXAÇÃO ──────────────────────────────────────────
        self.stimuli.show_fixation(
            elapsed=elapsed(),
            trials_done=trials_done
        )
        core.wait(self.config["fixation_time"])

        if "space" in event.getKeys(keyList=["space"]):
            return False

        # ── CUE — mostra o símbolo, diz o que vem a seguir ──
        self.logger.add_marker(time.time(), "cue_on", label)

        self.stimuli.show_cue(
            symbol, class_name,
            elapsed=elapsed(),
            trials_done=trials_done
        )
        core.wait(self.config["cue_time"])

        # ── MOTOR IMAGERY — janela activa, símbolo amarelo ───
        self.logger.add_marker(time.time(), "mi_start", label)

        self.stimuli.show_imagery(
            symbol, class_name,
            elapsed=elapsed(),
            trials_done=trials_done
        )
        core.wait(self.config["imagery_time"])

        self.logger.add_marker(time.time(), "mi_end", label)

        # ── REPOUSO ───────────────────────────────────────────
        self.stimuli.show_rest(
            elapsed=elapsed(),
            trials_done=trials_done
        )
        core.wait(self.config["rest_time"])

        if "space" in event.getKeys(keyList=["space"]):
            return False

        return True

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        mins = MAX_TRAIN_SECONDS // 60

        self.stimuli.show_message(
            f"TREINO\n\n"
            f"Duração máxima: {mins} minutos\n\n"
            f"Segue as setas no ecrã e imagina o movimento.\n\n"
            f"Podes parar a qualquer momento com SPACE.\n\n"
            f"Press SPACE para começar"
        )
        event.waitKeys(keyList=["space"])

        self.baseline()

        self.stimuli.show_message("A começar o treino...\n\nPrepara-te.")
        core.wait(2)

        session_start  = time.time()
        trials_done    = 0
        stopped_early  = False

        while True:

            elapsed = time.time() - session_start

            if elapsed >= MAX_TRAIN_SECONDS:
                print(f"\nTempo de treino esgotado ({mins} min).")
                break

            class_name = self.next_class()
            trial_ok   = self.run_trial(class_name, trials_done, session_start)
            trials_done += 1

            if not trial_ok:
                stopped_early = True
                print(f"\nTreino parado pelo utilizador após {trials_done} trials.")
                break

        # ── RESUMO ────────────────────────────────────────────
        elapsed_total = time.time() - session_start
        mins_done     = int(elapsed_total // 60)
        secs_done     = int(elapsed_total % 60)

        class_counts = {c: 0 for c in self.class_names}
        for m in self.logger.markers:
            if m["event"] == "mi_start":
                for c in self.class_names:
                    if self.config["classes"][c]["label"] == m["label"]:
                        class_counts[c] += 1

        lines = [
            "TREINO CONCLUÍDO\n",
            f"Duração: {mins_done}m {secs_done}s",
            f"Trials:  {trials_done}\n",
        ]
        for c, n in class_counts.items():
            lines.append(f"{self.config['classes'][c]['symbol']}  {c}: {n}")

        if stopped_early:
            lines.append("\n(parado manualmente)")

        lines.append("\n\nA processar dados...")

        self.stimuli.show_message("\n".join(lines))
        core.wait(3)

        self.win.close()

        print("\n" + "=" * 40)
        print(f"Treino: {trials_done} trials em {mins_done}m{secs_done}s")
        for c, n in class_counts.items():
            print(f"  {c}: {n} trials")
        print("=" * 40)
