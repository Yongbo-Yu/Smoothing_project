(TeX-add-style-hook
 "smoothing_paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "leqno") ("xy" "all") ("hyperref" "pdftex" "dvips")))
   (TeX-run-style-hooks
    "amsmath"
    "amsthm"
    "amssymb"
    "graphicx"
    "calc"
    "mathtools"
    "enumitem"
    "ifpdf"
    "ifthen"
    "braket"
    "xcolor"
    "xy"
    "fancyhdr"
    "tikz"
    "todonotes"
    "dsfont"
    "subcaption"
    "sidecap"
    "hyperref")
   (TeX-add-symbols
    '("angbrack" 1)
    '("ceil" 1)
    '("floor" 1)
    '("rip" 2)
    '("ip" 2)
    '("indic" 1)
    '("norm" 1)
    '("abs" 1)
    '("pderiv" 1)
    '("f" 2)
    "R"
    "N"
    "Z"
    "F"
    "dom"
    "Rplus"
    "Ocal"
    "pa"
    "half"
    "dx"
    "dt"
    "const"
    "tol"
    "barX"
    "bartau"
    "bartheta"
    "barp"
    "barrho"
    "baru"
    "barx"
    "barlambda"
    "barH"
    "barkappa"
    "barXi"
    "barsigma"
    "tV"
    "tp"
    "tm"
    "tS"
    "fB"
    "fW"
    "fxi"
    "fomega"
    "Cbv"
    "Heis"
    "h"
    "dist"
    "argmin"
    "sign"
    "id"
    "cor"
    "var"
    "supp"
    "diag")
   (LaTeX-add-environments
    "theorem"
    "lemma"
    "proposition"
    "corollary"
    "algorithm"
    "definition"
    "example"
    "assumption"
    "remark"
    "question")))

