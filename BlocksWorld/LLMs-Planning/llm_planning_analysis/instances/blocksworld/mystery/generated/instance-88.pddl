(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d e i f h k g j)
(:init 
(harmony)
(planet d)
(planet e)
(planet i)
(planet f)
(planet h)
(planet k)
(planet g)
(planet j)
(province d)
(province e)
(province i)
(province f)
(province h)
(province k)
(province g)
(province j)
)
(:goal
(and
(craves d e)
(craves e i)
(craves i f)
(craves f h)
(craves h k)
(craves k g)
(craves g j)
)))