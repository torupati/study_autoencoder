import torch
#
# pytest tests/test_torch00.py  -v -s
#

def test_tensor_addition():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    result = x + y
    expected = torch.tensor([4.0, 6.0])
    assert torch.equal(result, expected)

def test_autograd():
    x = torch.randn(2, 2, requires_grad=True)
    y = torch.randn(2, 2, requires_grad=True)
    z = x ** 2 + y * x + y ** 2
    z.backward(torch.ones(2, 2))
    x_grad = 2 * x + y
    y_grad = x + 2 * y
    assert torch.equal(x.grad, x_grad)
    assert torch.equal(y.grad, y_grad)

