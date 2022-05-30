from prediction import prediction
from prediction import plot_preds

def test_prediction():
    preds = prediction('../BDD/revolver/1.jpg')
    assert 0<preds["revolver"]<1
    assert 0<preds["rifle"]<1
    assert 0<preds["assault_rifle"]<1
