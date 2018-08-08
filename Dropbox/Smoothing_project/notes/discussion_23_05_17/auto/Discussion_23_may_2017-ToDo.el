(TeX-add-style-hook
 "Discussion_23_may_2017-ToDo"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "smoothing_paper")
   (TeX-add-symbols
    '("required" 1)
    "SMALLSKIP"
    "MEDSKIP"
    "BIGSKIP")
   (LaTeX-add-labels
    "Brownian_bridge"
    "lognormal_dynamics"
    "xact_location_continuous_problem"
    "Discrete_problem"
    "polynomial_kink_location"
    "polynomial_kink_location_derivative"
    "sec:choice-functional"
    "sec:plan-work-misc"
    "sec:smoothing")))

