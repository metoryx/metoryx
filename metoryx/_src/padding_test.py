from .padding import canonicalize_padding


class TestCanonicalizePadding:
    def test_integer(self):
        canonicalized = canonicalize_padding(1, (3, 3))
        assert canonicalized == [(1, 1), (1, 1)]

    def test_integers(self):
        canonicalized = canonicalize_padding([1, 2], (3, 3))
        assert canonicalized == [(1, 1), (2, 2)]

    def test_integer_tuples(self):
        canonicalized = canonicalize_padding([(1, 2), (3, 4)], (3, 3))
        assert canonicalized == [(1, 2), (3, 4)]

    def test_same(self):
        canonicalized = canonicalize_padding("SAME", (3, 3))
        assert canonicalized == "SAME"

    def test_valid(self):
        canonicalized = canonicalize_padding("VALID", (3, 3))
        assert canonicalized == "VALID"
