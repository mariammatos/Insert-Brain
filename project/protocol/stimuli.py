# ============================================================
# FILE: protocol/stimuli.py
# ============================================================

from psychopy import visual


class Stimuli:

    def __init__(self, win):

        self.win = win

        # ── Estímulos principais (iguais ao original) ────────

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

        # ── HUD — canto superior esquerdo ────────────────────
        # Tempo restante
        self.hud_time = visual.TextStim(
            win,
            text="",
            pos=(-0.75, 0.45),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="left"
        )

        # Trials feitos
        self.hud_trials = visual.TextStim(
            win,
            text="",
            pos=(-0.75, 0.39),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="left"
        )

        # ── HUD — canto superior direito ─────────────────────
        # Classe atual
        self.hud_class = visual.TextStim(
            win,
            text="",
            pos=(0.75, 0.45),
            height=0.04,
            color="#AAAAAA",
            anchorHoriz="right"
        )

        # ── Barra de progresso de tempo ──────────────────────
        # Fundo cinzento
        self.bar_bg = visual.Rect(
            win,
            width=1.4,
            height=0.025,
            pos=(0, -0.46),
            fillColor="#333333",
            lineColor=None
        )

        # Preenchimento verde (vai encolhendo)
        self.bar_fill = visual.Rect(
            win,
            width=1.4,
            height=0.025,
            pos=(0, -0.46),
            fillColor="#44BB77",
            lineColor=None
        )

        # ── Fase do trial (FIXAÇÃO / CUE / IMAGINA / REPOUSO) ─
        self.phase_text = visual.TextStim(
            win,
            text="",
            pos=(0, 0.38),
            height=0.04,
            color="#888888"
        )

        # Guarda o tempo total para calcular a barra
        self._max_seconds = 1.0

    # ========================================================
    # CONFIGURAR TEMPO MÁXIMO (chamar no início do protocolo)
    # ========================================================

    def set_max_time(self, max_seconds):
        self._max_seconds = max_seconds

    # ========================================================
    # ACTUALIZAR HUD
    # Chamar antes de qualquer draw, passa os valores actuais
    # ========================================================

    def update_hud(self, elapsed_seconds, trials_done, current_class=""):
        """
        Actualiza os elementos do HUD sem os desenhar ainda.
        elapsed_seconds : segundos desde o início do treino
        trials_done     : número de trials completos
        current_class   : nome da classe actual (ex: "LEFT")
        """

        remaining = max(0.0, self._max_seconds - elapsed_seconds)
        mins      = int(remaining // 60)
        secs      = int(remaining % 60)

        self.hud_time.text   = f"⏱  {mins}:{secs:02d}"
        self.hud_trials.text = f"✓  {trials_done} trials"
        self.hud_class.text  = current_class

        # Barra de progresso — encolhe da direita para a esquerda
        ratio     = max(0.0, remaining / self._max_seconds)
        bar_width = 1.4 * ratio
        self.bar_fill.width = max(0.001, bar_width)
        self.bar_fill.pos   = (-0.7 + bar_width / 2, -0.46)

        # Cor muda para laranja nos últimos 2 min, vermelho no último 1 min
        if remaining < 60:
            self.bar_fill.fillColor = "#DD4444"
        elif remaining < 120:
            self.bar_fill.fillColor = "#DDAA33"
        else:
            self.bar_fill.fillColor = "#44BB77"

    def _draw_hud(self):
        """Desenha todos os elementos do HUD."""
        self.bar_bg.draw()
        self.bar_fill.draw()
        self.hud_time.draw()
        self.hud_trials.draw()
        if self.hud_class.text:
            self.hud_class.draw()

    # ========================================================
    # MOSTRAR FASE DO TRIAL
    # ========================================================

    def _draw_phase(self, phase):
        """Desenha o texto da fase actual (FIXAÇÃO, IMAGINA, etc.)"""
        self.phase_text.text = phase
        self.phase_text.draw()

    # ========================================================
    # MÉTODOS PÚBLICOS — compatíveis com o original + HUD
    # ========================================================

    def show_fixation(self, elapsed=None, trials_done=None):
        """
        Cruz de fixação.
        Se elapsed e trials_done forem passados, mostra o HUD.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0)
            self._draw_hud()

        self._draw_phase("FIXAÇÃO")
        self.fixation.draw()
        self.win.flip()

    def show_cue(self, symbol, text, elapsed=None, trials_done=None):
        """
        Seta / símbolo de cue + texto da classe.
        Se elapsed e trials_done forem passados, mostra o HUD.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0, current_class=text)
            self._draw_hud()

        self._draw_phase("IMAGINA")
        self.cue.text  = symbol
        self.info.text = text
        self.cue.draw()
        self.info.draw()
        self.win.flip()

    def show_imagery(self, symbol, text, elapsed=None, trials_done=None):
        """
        Igual ao show_cue mas com fase IMAGINA em destaque.
        Usa cor mais viva para indicar que é a janela activa.
        """

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0, current_class=text)
            self._draw_hud()

        self._draw_phase("★  IMAGINA  ★")
        self.cue.text  = symbol
        self.info.text = text

        # Destaca o símbolo durante a janela de imagery
        original_color = self.cue.color
        self.cue.color = "#FFDD44"
        self.cue.draw()
        self.cue.color = original_color

        self.info.draw()
        self.win.flip()

    def show_rest(self, elapsed=None, trials_done=None):
        """Cruz de repouso entre trials com HUD."""

        if elapsed is not None:
            self.update_hud(elapsed, trials_done or 0)
            self._draw_hud()

        self._draw_phase("REPOUSO")
        self.fixation.draw()
        self.win.flip()

    def show_message(self, message):
        """Mensagem full-screen (igual ao original, sem HUD)."""

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
        Feedback breve entre trials:
        mostra qual foi a classe do trial que acabou.
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
