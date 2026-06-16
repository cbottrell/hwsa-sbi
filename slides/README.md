# LaTeX Slide Deck

The workshop slide deck is written in Beamer:

- Source: `sbi_workshop_deck.tex`
- Rendered PDF: `sbi_workshop_deck.pdf`
- Logo asset: `graphics/ICRARJointVenturePartners-black-background-colour-ICRAR.png`

Compile from the repository root with:

```bash
mkdir -p slides/build
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=slides/build slides/sbi_workshop_deck.tex
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=slides/build slides/sbi_workshop_deck.tex
cp slides/build/sbi_workshop_deck.pdf slides/sbi_workshop_deck.pdf
```

The deck uses a custom black, white, and crimson Beamer theme inspired by the
provided template slides.
