(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects g c f d a i j l)
(:init 
(handempty)
(ontable g)
(ontable c)
(ontable f)
(ontable d)
(ontable a)
(ontable i)
(ontable j)
(ontable l)
(clear g)
(clear c)
(clear f)
(clear d)
(clear a)
(clear i)
(clear j)
(clear l)
)
(:goal
(and
(on g c)
(on c f)
(on f d)
(on d a)
(on a i)
(on i j)
(on j l)
)))