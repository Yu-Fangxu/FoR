(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e f i b k)
(:init 
(harmony)
(planet e)
(planet f)
(planet i)
(planet b)
(planet k)
(province e)
(province f)
(province i)
(province b)
(province k)
)
(:goal
(and
(craves e f)
(craves f i)
(craves i b)
(craves b k)
)))