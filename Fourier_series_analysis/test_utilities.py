"""
MIT License
Copyright © 2024 David A. Reiss
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""


import pytest
from utilities import einsum_subscripts


@pytest.mark.parametrize("subs_1, subs_2, subs_3, expected_result", [
        ("alpha_0,beta'", "\gamma,beta'", "alpha_0,\gamma", "ab,cb->ac"),
        ("alpha,beta", "beta,alpha", "", "ab,ba->"),
        ("alpha,...", "...,alpha", "", "a...,...a->")
    ])
def test_einsum_subscripts(subs_1: str, subs_2: str, subs_3: str, expected_result: str):
    """
    Test function einsum_subscripts.
    """

    assert einsum_subscripts(subs_1, subs_2, final_indices=subs_3) == expected_result, \
        (f"expected result: {expected_result}, "
         f"actual result: {einsum_subscripts(subs_1, subs_2, final_indices=subs_3)}")
