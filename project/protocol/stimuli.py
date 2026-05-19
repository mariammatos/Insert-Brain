# ============================================================
# FILE: protocol/stimuli.py
# ============================================================

from psychopy import visual


class Stimuli:

    def __init__(self, win):

        self.win = win

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
            pos=(0, -0.5),
            height=0.05,
            color="white"
        )

    def show_fixation(self):

        self.fixation.draw()

        self.win.flip()

    def show_cue(self, symbol, text):

        self.cue.text = symbol

        self.info.text = text

        self.cue.draw()

        self.info.draw()

        self.win.flip()

    def show_message(self, message):

        msg = visual.TextStim(
            self.win,
            text=message,
            height=0.06,
            wrapWidth=1.5,
            color="white"
        )

        msg.draw()

        self.win.flip()