(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c i j e l d)
(:init 
(handempty)
(ontable c)
(ontable i)
(ontable j)
(ontable e)
(ontable l)
(ontable d)
(clear c)
(clear i)
(clear j)
(clear e)
(clear l)
(clear d)
)
(:goal
(and
(on c i)
(on i j)
(on j e)
(on e l)
(on l d)
)))