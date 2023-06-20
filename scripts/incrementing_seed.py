import gradio as gr

import modules.scripts as scripts
from modules.processing import StableDiffusionProcessing
from modules.scripts import PostprocessImageArgs, AlwaysVisible


class Script(scripts.Script):
    def title(self):
        return "Incrementing seed"

    def show(self, is_img2img):
        """This method MUST return AlwaysVisible for #postprocess_image to be called"""
        return AlwaysVisible

    def ui(self, is_img2img):
        inc_enabled = gr.Checkbox(
            label="Enable Incrementing Seed",
            value=True,
            visible=True,
        )

        return [inc_enabled]

    def postprocess_image(
        self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, *args
    ):
        (inc_enabled,) = args
        if not inc_enabled:
            return

        p.seed += 1
        p.seeds = [x + 1 for x in p.seeds]
        p.subseeds = [x + 1 for x in p.subseeds]
        p.all_seeds = [x + 1 for x in p.all_seeds]
        p.all_subseeds = [x + 1 for x in p.all_subseeds]
