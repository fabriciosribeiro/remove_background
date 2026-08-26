import gradio as gr
from transformers import pipeline


# Carrega o modelo uma única vez
pipe = pipeline(
    "image-segmentation",
    model="briaai/RMBG-1.4",
    trust_remote_code=True
)


def remove_background(image):
    """
    Remove automaticamente o fundo de uma imagem
    utilizando o modelo BRIA RMBG-1.4.
    """
    return pipe(image)


app = gr.Interface(
    fn=remove_background,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil", format="png"),
    title="Remoção de Background de Imagens",
    description=(
        "Envie uma imagem e veja o background removido "
        "automaticamente. A imagem resultante será no formato PNG."
    )
)


if __name__ == "__main__":
    app.launch()
