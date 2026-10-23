"""Generates build_paper_inderscience.py: the Inderscience layout (_infra_inderscience.py) plus the unchanged
manuscript body taken from build_paper.py (Introduction up to, not including, the Biographical notes)."""
import os

H = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(H, "build_paper.py"), encoding="utf-8").read()
a = src.index("# ============================================================= 1 Introduction")
b = src.index('h1("Biographical notes")')
body = src[a:b]
body = body.replace('p = para(r, size=10, after=3, align="l")', 'p = para(r, size=9, after=3, align="j", first=False)')
infra = open(os.path.join(H, "_infra_inderscience.py"), encoding="utf-8").read()
tail = '''
out = os.path.join(HERE, "IJCC_Manuscript_Inderscience_Layout.docx")
doc.save(out)
print("saved", out, "tables", TABLES[0], "figures", FIGURES[0])
'''
open(os.path.join(H, "build_paper_inderscience.py"), "w", encoding="utf-8").write(infra + body + tail)
