#import "@preview/charged-ieee:0.1.2": ieee

#show: ieee.with(
  title: [Research proposal],
  abstract: [
    #lorem(50)
  ],
  authors: (
    (
      name: "David van Munster",
      department: [Data Science],
      organization: [Radboud University],
      location: [Nijmegen, The Netherlands],
      email: "david.vanmunster@ru.nl",
    ),
  ),
  index-terms: (),
  bibliography: bibliography("refs.bib"),
)

= Introduction
Scientific writing is a crucial part of the research process, allowing researchers to share their findings with the wider scientific community. However, the process of typesetting scientific documents can often be a frustrating and time-consuming affair, particularly when using outdated tools such as LaTeX. Despite being over 30 years old, it remains a popular choice for scientific writing due to its power and flexibility. However, it also comes with a steep learning curve, complex syntax, and long compile times, leading to frustration and despair for many researchers. @netwok2020
= Methods
#lorem(90)

$ a + b = gamma $

#lorem(200)
