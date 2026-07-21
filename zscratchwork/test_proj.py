import numpy as np

from src.perturbations import projection_distance

def test_basic_span():
    fpa1 = [1,0,0,0]
    fpa2 = [0,1,0,0]
    
    fpa3 = [1,1,0,0]
    fpa4 = [0,0,0,1]

    dist1 = projection_distance(fpa1,fpa2,fpa3)
    dist2 = projection_distance(fpa1,fpa2,fpa4)
    assert dist1 == 0 and dist2 == 1

def test_degenerate_span():
    fpa1 = [1,0,0,0]
    fpa2 = [2,0,0,0]
    
    fpa3 = [3,0,0,0]
    fpa4 = [1,1,0,0]

    dist1 = projection_distance(fpa1,fpa2,fpa3)
    dist2 = projection_distance(fpa1,fpa2,fpa4)
    assert dist1 == 0 and dist2 == 1