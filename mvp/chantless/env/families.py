"""Procedural Python function task families for the codetasks environment.

Each family is a parameterised generator. A variant fixes the parameters, the function name and
the hidden tests. Reference solutions run only inside the trusted generator; the model never
sees them. Splits are assigned per family, so held-out splits test unseen task *kinds*.
"""
from __future__ import annotations

import random
import string


def _words(rng, n, lo=2, hi=7):
    return [''.join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi))) for _ in range(n)]


def _ints(rng, n, lo=-20, hi=20):
    return [rng.randint(lo, hi) for _ in range(n)]


def _names(rng, *opts):
    return rng.choice(opts)


FAMILIES: dict[str, callable] = {}


def family(fn):
    FAMILIES[fn.__name__] = fn
    return fn


# Each family returns (fname, signature, description, reference_source, list_of_arg_tuples)

@family
def sum_multiples(rng):
    k = rng.randint(2, 5)
    f = _names(rng, "sum_multiples", "total_of_multiples", "add_multiples")
    ref = f"def {f}(nums):\n    return sum(x for x in nums if x % {k} == 0)\n"
    return f, "(nums: list[int]) -> int", f"returns the sum of the numbers in `nums` that are divisible by {k}", ref, \
        [(_ints(rng, rng.randint(0, 9)),) for _ in range(12)]


@family
def count_char(rng):
    c = rng.choice("aeiost")
    f = _names(rng, "count_char", "occurrences", "char_count")
    ref = f"def {f}(s):\n    return s.lower().count('{c}')\n"
    return f, "(s: str) -> int", f"returns how many times the letter '{c}' appears in `s`, ignoring case", ref, \
        [(' '.join(_words(rng, rng.randint(0, 5))).upper() if rng.random() < .3 else ' '.join(_words(rng, rng.randint(0, 5))),) for _ in range(12)]


@family
def reverse_words(rng):
    sep = rng.choice([" ", "-"])
    f = _names(rng, "reverse_words", "flip_words", "words_backwards")
    ref = f"def {f}(s):\n    return {sep!r}.join(reversed(s.split({sep!r})))\n"
    return f, "(s: str) -> str", f"returns `s` with the order of its {sep!r}-separated words reversed (split and join on {sep!r} exactly)", ref, \
        [(sep.join(_words(rng, rng.randint(1, 6))),) for _ in range(12)]


@family
def clamp_list(rng):
    lo = rng.randint(-10, 0); hi = rng.randint(1, 10)
    f = _names(rng, "clamp_all", "clip_values", "bound_list")
    ref = f"def {f}(nums):\n    return [min(max(x, {lo}), {hi}) for x in nums]\n"
    return f, "(nums: list[int]) -> list[int]", f"returns a new list where every number is limited to the range [{lo}, {hi}] (values below {lo} become {lo}, values above {hi} become {hi})", ref, \
        [(_ints(rng, rng.randint(0, 8)),) for _ in range(12)]


@family
def dedupe(rng):
    f = _names(rng, "dedupe", "unique_in_order", "remove_duplicates")
    ref = f"def {f}(items):\n    seen = set(); out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out\n"
    return f, "(items: list[int]) -> list[int]", "returns the items with duplicates removed, keeping the first occurrence of each and preserving order", ref, \
        [(_ints(rng, rng.randint(0, 10), 0, 5),) for _ in range(12)]


@family
def run_length(rng):
    f = _names(rng, "run_length", "rle_encode", "compress_runs")
    ref = (f"def {f}(s):\n    out = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
           f"        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)\n")
    return f, "(s: str) -> str", "returns the run-length encoding of `s`: each run of a repeated character becomes the character followed by the run length, e.g. 'aaab' -> 'a3b1'", ref, \
        [(''.join(rng.choice("abc") * rng.randint(1, 3) for _ in range(rng.randint(0, 5))),) for _ in range(12)]


@family
def caesar(rng):
    k = rng.randint(1, 5)
    f = _names(rng, "caesar", "shift_letters", "rotate_letters")
    ref = (f"def {f}(s):\n    out = []\n    for ch in s:\n        if 'a' <= ch <= 'z':\n            out.append(chr((ord(ch) - 97 + {k}) % 26 + 97))\n"
           f"        else:\n            out.append(ch)\n    return ''.join(out)\n")
    return f, "(s: str) -> str", f"shifts every lowercase letter in `s` forward by {k} places in the alphabet (wrapping z to a) and leaves all other characters unchanged", ref, \
        [(' '.join(_words(rng, rng.randint(1, 4))) + rng.choice(["", "!", " 42"]),) for _ in range(12)]


@family
def fib(rng):
    f = _names(rng, "fib", "fibonacci", "nth_fib")
    a, b = rng.choice([(0, 1), (1, 1), (2, 1)])
    ref = f"def {f}(n):\n    a, b = {a}, {b}\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
    return f, "(n: int) -> int", f"returns term n (0-indexed) of the sequence that starts {a}, {b} and where each later term is the sum of the previous two", ref, \
        [(rng.randint(0, 25),) for _ in range(12)]


@family
def chunk(rng):
    k = rng.randint(2, 4)
    f = _names(rng, "chunk", "split_into_groups", "batch_list")
    ref = f"def {f}(items):\n    return [items[i:i + {k}] for i in range(0, len(items), {k})]\n"
    return f, "(items: list[int]) -> list[list[int]]", f"splits `items` into consecutive groups of {k} (the last group may be shorter)", ref, \
        [(_ints(rng, rng.randint(0, 9)),) for _ in range(12)]


@family
def rotate(rng):
    k = rng.randint(1, 3)
    d = rng.choice(["left", "right"])
    f = _names(rng, "rotate", "rotate_list", "shift_list")
    body = f"items[{k} % len(items):] + items[:{k} % len(items)]" if d == "left" else f"items[-({k} % len(items)):] + items[:-({k} % len(items))] if {k} % len(items) else items[:]"
    ref = f"def {f}(items):\n    if not items:\n        return []\n    return {body}\n"
    return f, "(items: list[int]) -> list[int]", f"returns `items` rotated {d} by {k} positions (an empty list stays empty)", ref, \
        [(_ints(rng, rng.randint(0, 7)),) for _ in range(12)]


@family
def second_largest(rng):
    f = _names(rng, "second_largest", "runner_up", "second_max")
    ref = f"def {f}(nums):\n    u = sorted(set(nums))\n    return u[-2] if len(u) >= 2 else None\n"
    return f, "(nums: list[int]) -> int | None", "returns the second largest distinct value in `nums`, or None if there are fewer than two distinct values", ref, \
        [(_ints(rng, rng.randint(0, 7), -5, 5),) for _ in range(12)]


@family
def word_freq(rng):
    f = _names(rng, "word_freq", "count_words", "word_counts")
    ref = f"def {f}(s):\n    out = {{}}\n    for w in s.lower().split():\n        out[w] = out.get(w, 0) + 1\n    return out\n"
    return f, "(s: str) -> dict[str, int]", "returns a dict mapping each lowercase word in `s` (split on whitespace) to the number of times it occurs", ref, \
        [(' '.join(rng.choice(["a", "b", "cat", "Dog", "dog", "x"]) for _ in range(rng.randint(0, 8))),) for _ in range(12)]


@family
def digit_sum(rng):
    f = _names(rng, "digit_sum", "sum_of_digits", "add_digits")
    ref = f"def {f}(n):\n    return sum(int(c) for c in str(abs(n)))\n"
    return f, "(n: int) -> int", "returns the sum of the decimal digits of `n` (ignore the sign)", ref, \
        [(rng.randint(-99999, 99999),) for _ in range(12)]


@family
def is_prime(rng):
    f = _names(rng, "is_prime", "prime_check", "check_prime")
    ref = f"def {f}(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n"
    return f, "(n: int) -> bool", "returns True if `n` is a prime number and False otherwise", ref, \
        [(rng.randint(-3, 200),) for _ in range(12)]


@family
def to_binary(rng):
    w = rng.randint(4, 8)
    f = _names(rng, "to_binary", "binary_string", "bits")
    ref = f"def {f}(n):\n    return format(n, '0{w}b')\n"
    return f, "(n: int) -> str", f"returns the binary representation of the non-negative integer `n` as a string, left-padded with zeros to at least {w} characters", ref, \
        [(rng.randint(0, 300),) for _ in range(12)]


@family
def long_words(rng):
    k = rng.randint(2, 5)
    f = _names(rng, "count_long_words", "long_word_count", "words_longer")
    ref = f"def {f}(s):\n    return sum(1 for w in s.split() if len(w) > {k})\n"
    return f, "(s: str) -> int", f"returns how many whitespace-separated words in `s` have more than {k} characters", ref, \
        [(' '.join(_words(rng, rng.randint(0, 7), 1, 8)),) for _ in range(12)]


@family
def merge_sorted(rng):
    f = _names(rng, "merge_sorted", "merge_lists", "combine_sorted")
    ref = f"def {f}(a, b):\n    return sorted(a + b)\n"
    return f, "(a: list[int], b: list[int]) -> list[int]", "merges two already-sorted lists into one sorted list", ref, \
        [(sorted(_ints(rng, rng.randint(0, 5))), sorted(_ints(rng, rng.randint(0, 5)))) for _ in range(12)]


@family
def transpose(rng):
    f = _names(rng, "transpose", "flip_matrix", "matrix_transpose")
    ref = f"def {f}(m):\n    return [list(r) for r in zip(*m)]\n"
    def mk():
        r, c = rng.randint(1, 4), rng.randint(1, 4)
        return ([[rng.randint(0, 9) for _ in range(c)] for _ in range(r)],)
    return f, "(m: list[list[int]]) -> list[list[int]]", "returns the transpose of the rectangular matrix `m` (rows become columns)", ref, [mk() for _ in range(12)]


@family
def anagram(rng):
    f = _names(rng, "is_anagram", "anagrams", "same_letters")
    ref = f"def {f}(a, b):\n    return sorted(a.replace(' ', '').lower()) == sorted(b.replace(' ', '').lower())\n"
    def mk():
        w = ''.join(_words(rng, 1, 3, 6))
        b = list(w); rng.shuffle(b)
        b = ''.join(b) if rng.random() < .5 else ''.join(_words(rng, 1, 3, 6))
        return (w, b.upper() if rng.random() < .3 else b)
    return f, "(a: str, b: str) -> bool", "returns True if `a` and `b` are anagrams of each other, ignoring spaces and letter case", ref, [mk() for _ in range(12)]


@family
def common_prefix(rng):
    f = _names(rng, "common_prefix", "longest_prefix", "shared_prefix")
    ref = (f"def {f}(words):\n    if not words:\n        return ''\n    p = words[0]\n    for w in words[1:]:\n        while not w.startswith(p):\n            p = p[:-1]\n    return p\n")
    def mk():
        base = ''.join(_words(rng, 1, 0, 3))
        return ([base + w for w in _words(rng, rng.randint(1, 4), 0, 3)],)
    return f, "(words: list[str]) -> str", "returns the longest common prefix of all strings in `words` ('' for an empty list)", ref, [mk() for _ in range(12)]


@family
def balanced(rng):
    pairs = rng.choice(["()", "()[]", "()[]{}"])
    f = _names(rng, "is_balanced", "balanced", "brackets_ok")
    opens, closes = pairs[0::2], pairs[1::2]
    ref = (f"def {f}(s):\n    st = []\n    m = dict(zip({closes!r}, {opens!r}))\n    for ch in s:\n        if ch in {opens!r}:\n            st.append(ch)\n"
           f"        elif ch in {closes!r}:\n            if not st or st.pop() != m[ch]:\n                return False\n    return not st\n")
    return f, "(s: str) -> bool", f"returns True if the brackets {' '.join(pairs)} in `s` are balanced and properly nested (other characters are ignored)", ref, \
        [(''.join(rng.choice(pairs + "a") for _ in range(rng.randint(0, 8))),) for _ in range(12)]


@family
def running_sum(rng):
    f = _names(rng, "running_sum", "cumulative_sum", "prefix_sums")
    ref = f"def {f}(nums):\n    out, t = [], 0\n    for x in nums:\n        t += x\n        out.append(t)\n    return out\n"
    return f, "(nums: list[int]) -> list[int]", "returns the running (cumulative) sums of `nums`", ref, [(_ints(rng, rng.randint(0, 8)),) for _ in range(12)]


@family
def diffs(rng):
    f = _names(rng, "differences", "pairwise_diffs", "gaps")
    ref = f"def {f}(nums):\n    return [b - a for a, b in zip(nums, nums[1:])]\n"
    return f, "(nums: list[int]) -> list[int]", "returns the differences between consecutive elements (next minus current); a list with fewer than two elements gives []", ref, \
        [(_ints(rng, rng.randint(0, 8)),) for _ in range(12)]


@family
def prefix_filter(rng):
    p = rng.choice(["a", "b", "co", "s"])
    f = _names(rng, "starts_with", "filter_prefix", "keep_prefixed")
    ref = f"def {f}(words):\n    return [w for w in words if w.startswith({p!r})]\n"
    return f, "(words: list[str]) -> list[str]", f"returns the words that start with {p!r}, in their original order", ref, \
        [([rng.choice(["", p, p[0]]) + w for w in _words(rng, rng.randint(0, 6), 1, 4)],) for _ in range(12)]


@family
def snake_to_camel(rng):
    f = _names(rng, "snake_to_camel", "to_camel", "camelize")
    ref = f"def {f}(s):\n    parts = s.split('_')\n    return parts[0] + ''.join(p.capitalize() for p in parts[1:])\n"
    return f, "(s: str) -> str", "converts a snake_case identifier to lowerCamelCase (e.g. 'max_value_x' -> 'maxValueX')", ref, \
        [('_'.join(_words(rng, rng.randint(1, 4))),) for _ in range(12)]


@family
def count_upper(rng):
    f = _names(rng, "count_upper", "uppercase_count", "capitals")
    ref = f"def {f}(s):\n    return sum(1 for c in s if c.isupper())\n"
    return f, "(s: str) -> int", "returns the number of uppercase letters in `s`", ref, \
        [(''.join(rng.choice(string.ascii_letters + " 1") for _ in range(rng.randint(0, 15))),) for _ in range(12)]


@family
def roman(rng):
    f = _names(rng, "to_roman", "roman_numeral", "int_to_roman")
    ref = (f"def {f}(n):\n    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n"
           f"    out = ''\n    for v, s in vals:\n        while n >= v:\n            out += s\n            n -= v\n    return out\n")
    return f, "(n: int) -> str", "converts an integer 1..3999 to a Roman numeral using standard subtractive notation", ref, [(rng.randint(1, 3999),) for _ in range(12)]


@family
def leap(rng):
    f = _names(rng, "is_leap", "leap_year", "check_leap")
    ref = f"def {f}(y):\n    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)\n"
    return f, "(y: int) -> bool", "returns True if year `y` is a Gregorian leap year", ref, [(rng.choice([1900, 2000, 2024, 2023, 2100, 2400]) + rng.choice([0, 0, 1, 4]),) for _ in range(12)]


@family
def triangle(rng):
    f = _names(rng, "triangle_kind", "classify_triangle", "triangle_type")
    ref = (f"def {f}(a, b, c):\n    x, y, z = sorted((a, b, c))\n    if x <= 0 or x + y <= z:\n        return 'invalid'\n"
           f"    if x == z:\n        return 'equilateral'\n    if x == y or y == z:\n        return 'isosceles'\n    return 'scalene'\n")
    return f, "(a: int, b: int, c: int) -> str", "returns 'equilateral', 'isosceles', 'scalene' or 'invalid' for the side lengths a, b, c (non-positive sides or a + b <= c for the two shortest sides is 'invalid')", ref, \
        [tuple(rng.randint(0, 6) for _ in range(3)) for _ in range(12)]


@family
def window_max(rng):
    k = rng.randint(2, 3)
    f = _names(rng, "window_max", "sliding_max", "max_windows")
    ref = f"def {f}(nums):\n    return [max(nums[i:i + {k}]) for i in range(len(nums) - {k} + 1)]\n"
    return f, "(nums: list[int]) -> list[int]", f"returns the maximum of every contiguous window of length {k} in `nums`, left to right ([] if the list is shorter than {k})", ref, \
        [(_ints(rng, rng.randint(0, 8)),) for _ in range(12)]


@family
def vowel_strip(rng):
    f = _names(rng, "strip_vowels", "remove_vowels", "devowel")
    ref = f"def {f}(s):\n    return ''.join(c for c in s if c.lower() not in 'aeiou')\n"
    return f, "(s: str) -> str", "returns `s` with every vowel (a, e, i, o, u in either case) removed", ref, \
        [(' '.join(_words(rng, rng.randint(0, 4))).title(),) for _ in range(12)]


@family
def gcd_list(rng):
    f = _names(rng, "gcd_all", "list_gcd", "common_divisor")
    ref = f"def {f}(nums):\n    import math\n    g = 0\n    for x in nums:\n        g = math.gcd(g, x)\n    return g\n"
    return f, "(nums: list[int]) -> int", "returns the greatest common divisor of all numbers in `nums` (0 for an empty list)", ref, \
        [([rng.choice([2, 3, 5, 6]) * rng.randint(1, 12) for _ in range(rng.randint(0, 4))],) for _ in range(12)]


@family
def mean_round(rng):
    d = rng.randint(0, 2)
    f = _names(rng, "average", "mean_of", "avg")
    ref = f"def {f}(nums):\n    return round(sum(nums) / len(nums), {d}) if nums else 0.0\n"
    return f, "(nums: list[int]) -> float", f"returns the mean of `nums` rounded with round(x, {d}), or 0.0 for an empty list", ref, \
        [(_ints(rng, rng.randint(0, 6), 0, 50),) for _ in range(12)]


@family
def flatten(rng):
    f = _names(rng, "flatten", "flatten_once", "concat_lists")
    ref = f"def {f}(lists):\n    return [x for sub in lists for x in sub]\n"
    return f, "(lists: list[list[int]]) -> list[int]", "flattens a list of lists by one level, preserving order", ref, \
        [([_ints(rng, rng.randint(0, 3)) for _ in range(rng.randint(0, 4))],) for _ in range(12)]


@family
def palindrome(rng):
    f = _names(rng, "is_palindrome", "palindrome", "reads_same")
    ref = f"def {f}(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n"
    def mk():
        w = ''.join(_words(rng, 1, 1, 4))
        return ((w + w[::-1]).title() + rng.choice(["", "!", " "]) if rng.random() < .5 else ' '.join(_words(rng, 2)),)
    return f, "(s: str) -> bool", "returns True if `s` reads the same forwards and backwards when only letters and digits are considered, ignoring case", ref, [mk() for _ in range(12)]


@family
def fizz(rng):
    a, b = rng.choice([(3, 5), (2, 7), (4, 6)])
    f = _names(rng, "fizzbuzz", "fizz_word", "fb")
    ref = (f"def {f}(n):\n    if n % {a} == 0 and n % {b} == 0:\n        return 'FizzBuzz'\n    if n % {a} == 0:\n        return 'Fizz'\n"
           f"    if n % {b} == 0:\n        return 'Buzz'\n    return str(n)\n")
    return f, "(n: int) -> str", f"returns 'FizzBuzz' if `n` is divisible by both {a} and {b}, 'Fizz' if only by {a}, 'Buzz' if only by {b}, otherwise str(n)", ref, \
        [(rng.randint(1, 60),) for _ in range(12)]


@family
def every_kth(rng):
    k = rng.randint(2, 4)
    f = _names(rng, "every_kth", "take_every", "stride")
    ref = f"def {f}(items):\n    return items[::{k}]\n"
    return f, "(items: list[int]) -> list[int]", f"returns every {k}th element of `items` starting with the first (indices 0, {k}, {2*k}, ...)", ref, \
        [(_ints(rng, rng.randint(0, 10)),) for _ in range(12)]


@family
def char_histogram(rng):
    f = _names(rng, "most_common_char", "top_char", "frequent_char")
    ref = (f"def {f}(s):\n    if not s:\n        return ''\n    best = None\n    for c in sorted(set(s)):\n        if best is None or s.count(c) > s.count(best):\n            best = c\n    return best\n")
    return f, "(s: str) -> str", "returns the character that occurs most often in `s`; break ties by choosing the alphabetically smallest; '' for an empty string", ref, \
        [(''.join(rng.choice("abcd") for _ in range(rng.randint(0, 9))),) for _ in range(12)]


@family
def count_pairs_sum(rng):
    t = rng.randint(3, 9)
    f = _names(rng, "count_pairs", "pair_count", "pairs_summing")
    ref = f"def {f}(nums):\n    n = len(nums)\n    return sum(1 for i in range(n) for j in range(i + 1, n) if nums[i] + nums[j] == {t})\n"
    return f, "(nums: list[int]) -> int", f"returns how many index pairs i < j satisfy nums[i] + nums[j] == {t}", ref, \
        [(_ints(rng, rng.randint(0, 7), 0, 9),) for _ in range(12)]


@family
def title_case(rng):
    f = _names(rng, "title_words", "capitalize_words", "titlecase")
    ref = f"def {f}(s):\n    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split(' '))\n"
    return f, "(s: str) -> str", "capitalises the first letter of each space-separated word and lowercases the rest (split and join on single spaces exactly)", ref, \
        [(' '.join(w.upper() if rng.random() < .3 else w for w in _words(rng, rng.randint(1, 4))),) for _ in range(12)]


@family
def max_gap(rng):
    f = _names(rng, "max_gap", "largest_gap", "widest_gap")
    ref = f"def {f}(nums):\n    s = sorted(nums)\n    return max((b - a for a, b in zip(s, s[1:])), default=0)\n"
    return f, "(nums: list[int]) -> int", "returns the largest difference between consecutive values after sorting `nums` (0 if fewer than two values)", ref, \
        [(_ints(rng, rng.randint(0, 7)),) for _ in range(12)]


@family
def digits_words(rng):
    f = _names(rng, "spell_digits", "digits_to_words", "say_digits")
    ref = (f"def {f}(n):\n    names = ['zero','one','two','three','four','five','six','seven','eight','nine']\n"
           f"    return ' '.join(names[int(c)] for c in str(n))\n")
    return f, "(n: int) -> str", "spells out each decimal digit of the non-negative integer `n` as an English word, separated by single spaces (e.g. 405 -> 'four zero five')", ref, \
        [(rng.randint(0, 99999),) for _ in range(12)]
