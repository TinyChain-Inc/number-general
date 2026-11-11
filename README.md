# number-general
A generic Rust number type with support for basic math operations, (de)serialization, and casting with
[safecast](https://github.com/TinyChain-Inc/safecast).

Generic text parsing delegates to the existing signed, unsigned, floating-point,
complex, and Boolean parsers in that order. Signed integers retain precedence;
wide unsigned values, complex scientific notation, and pure imaginary forms are
accepted without a separate token-classification heuristic.

Example usage:
```rust
use number_general::{Int, Number};
use safecast::CastFrom;

let sequence: Vec<Number> = serde_json::from_str("[true, 2, 3.5, -4, [1.0, -0.5]]").unwrap();
let actual = sequence.into_iter().product();

assert_eq!(actual, Number::from(num::Complex::<f64>::new(-28., 14.)));
assert_eq!(Int::cast_from(actual), Int::from(-28));
```

Numeric equality and comparison preserve exact values across representations;
arithmetic retains its dtype-promotion rules. Signed zero compares equally and
NaNs use a canonical equality/hash representation. Standard hashes agree for
equal numbers. Primitive cryptographic encodings remain type-specific.

Complex magnitudes retain their component precision and use a stable norm.
A complex number is false exactly when both components are zero; NaNs are truthy.

Collation orders all real-valued numbers numerically before non-real complex
numbers. Non-real complex values use a double-precision norm; distinct values may
collate equally. JSON may spell an f32 with its full exactly represented value.
These corrections are unreleased ordering changes: recreate development stores
whose indexes relied on the earlier comparisons rather than reopening them under
new semantics.

## Scalar arithmetic

Signed and unsigned integer operations promote to the widest operand in their
family. Addition, subtraction, multiplication, and signed absolute value wrap at
that width in every build profile. Signed minimum absolute value remains minimum.
Comparison and hashing preserve exact values and do not wrap; their shared
exclusive bound is derived from the widest supported integer type.

`NumberInstance::try_div` and `try_rem` return the existing `Error` for integer
zero divisors; Boolean division by false also returns an error. Nonzero integer
division truncates toward zero, signed minimum divided by -1 wraps, and remainder
has the dividend's sign. Boolean remainder remains XOR. Floating and complex
division retain their IEEE behavior. Native `/`, `%`, `/=`, and `%=` conveniences
use the same implementation and panic on returned errors; application execution
should use the fallible methods. Custom `NumberInstance` implementations must
provide both methods; `Error::new` constructs their recoverable errors.

Integer-result powers retain the base width and consume full-width integral
exponents without floating conversion. Negative integral exponents yield one for
base one, the parity result for signed base minus one, and zero otherwise.
Dynamic `Number::pow` keeps its existing floating promotion for negative integer
exponents, so an integer scalar base two raised to minus one produces 0.5.

ha-ndarray deliberately keeps its fixed-dtype policy: integer zero divisors
produce zero, and negative integer powers retain the array dtype. Its comparison
predicates use IEEE unordered NaNs; scalar equality and key collation retain
canonical NaN identity. These are distinct contracts, not implicit conversions.
Casts, encodings, and key ordering are unchanged by the arithmetic corrections.
