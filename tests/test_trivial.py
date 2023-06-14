from mom6_regional import parabola

def func(x):
    return x + 1

def test_answer():
    assert func(4) == 5

def test_parabola():
    assert parabola(2) == 4
