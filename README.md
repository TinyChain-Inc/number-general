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

## Numerical contracts

Choose the surface that fits the caller:

| Surface | Behavior |
| --- | --- |
| Ordinary operators and methods | Floating/complex IEEE behavior; integer overflow follows the application's Rust overflow settings |
| `try_*` methods | Structured errors for overflow, underflow, invalid domains, and unrepresentable results |
| `wrapping_*` methods | Explicit integer wrapping with ordinary promotion and floating/complex IEEE behavior |

Ordinary same-family integer operations promote widths before primitive Rust
arithmetic. Mixed signed/unsigned operations instead check their widened
mathematical result and panic if it cannot fit the selected destination, in every
profile. Assignments calculate the result before replacing their receiver.
TinyChain request execution uses checked operations; it does not rely on native
panics for arithmetic validation.

Promotion follows Boolean → unsigned → signed → float → complex, taking the
maximum width after family conversion. Existing unsigned-to-signed mappings are
retained, including U8→I16; Boolean→Int also uses I16. Mixed integer arithmetic
checks the result after evaluation, allowing cancellation even when an operand
alone cannot fit the destination.

### Checked operations

`NumberInstance` provides `try_add`, `try_sub`, `try_mul`, `try_div`, `try_rem`,
`try_abs`, `try_exp`, `try_ln`, `try_log`, `try_pow`, `try_round`, `try_into_type`,
`try_sum`, and `try_product`. `Trigonometry` has corresponding checked methods;
`NumberClass::try_cast` checks conversions. Custom implementations supply both
ordinary and checked numerical methods, without an additional policy trait.

Errors expose `Error::kind()`: `InvalidInput`, `Overflow`, `Underflow`,
`DivisionByZero`, `Domain`, `Unrepresentable`, or `Unsupported`.
`Error::with_kind` constructs a classified error; `Error::new` retains its existing
input-error diagnostic behavior.

Checked integer arithmetic uses widened intermediates and checks the selected
destination. Above the maximum is overflow; below the minimum is underflow.
Division truncates toward zero. Signed minimum divided by minus one errors; its
checked remainder is representable zero.

For finite floating and complex inputs, checked operations reject invalid domains,
new nonfinite results, and nonzero values lost to zero. Representable subnormals
and ordinary precision rounding are allowed. Cancellation, intentional rounding to
zero, and other defined zero results succeed. Checked real operations and retained
complex arithmetic preserve IEEE propagation from explicit NaN/infinity inputs.

Checked real transcendental functions, rounding and complex magnitude use the
pinned `libm` 0.2.16 software kernels, with
`force-soft-floats` overriding architecture/intrinsic features even when enabled
by another dependency. Ordinary and floating/complex wrapping methods retain
native Rust/backend math. No runtime backend selection or fallback is performed. Basic arithmetic uses
fixed-width Rust primitives under the IEEE environment described below.

For this implementation and a normal IEEE floating-point environment (nearest-even
arithmetic, subnormals enabled, no fast-math transformations), the checked contract
targets matching finite result bits, dtypes, signed zeros and error kinds across
validated builds/targets. NaN payloads are unspecified. This is not a promise for
arbitrary compiler/library versions or a modified floating-point environment.
Kernel/compiler upgrades must rerun the portability vectors and accuracy
references before extending the validated configuration. Custom numeric
implementations remain responsible for their own methods.
Conversions to `usize` are explicitly pointer-sized and outside that contract.

### Certified checked complex operations

| Operation | Supported inputs |
| --- | --- |
| Arithmetic, magnitude, rounding | Existing general complex support, including IEEE nonfinite arithmetic; exact cancellation checks are retained |
| Power | Bounded integral real exponents, including integral floats; complex and nonintegral exponents return `Unsupported` |
| Exponential | Finite real axis |
| Natural logarithm | Finite nonzero coordinate axes, preserving signed-zero branch sides |
| Logarithm | Those axes with a finite positive real base other than one |
| sin/cos/tan, sinh/cosh/tanh, asin/acos/atan, asinh/atanh | Existing finite real/imaginary-axis identities within their real-function domains; imaginary atan requires magnitude below one, and imaginary asinh requires magnitude at most one |
| acosh | Finite real axis at or above one |

The real domains for asin/acos are [-1, 1], for atanh (-1, 1), and for acosh
[1, infinity). Known zero/pole domain failures remain errors. Other complex
transcendental inputs, including nonfinite inputs, return `Unsupported` before
numerical evaluation. Complex-tagged logarithm bases and exponents are
unsupported even when their imaginary component is zero. Ordinary complex
methods remain available; checked methods
never retry through them. Auxiliary axis signs also use portable kernels.

Complex multiplication, division and remainder retain exact rational checks for
ambiguous zero/nonfinite components. That exceptional path allocates bounded
arbitrary-precision temporaries; general transcendental result guessing is removed.

Checked casts reject out-of-range values, lost fractions or imaginary components,
and Boolean destinations other than numeric zero or one. Finite precision rounding
is allowed; narrowing overflow and nonzero-to-zero underflow are errors.
`CastFrom`, `NumberClass::cast`, and `into_type` remain explicitly lossy conversions.
All four conversion surfaces resolve wildcard widths from the source.

### Powers, rounding, and aggregates

Integer-result powers retain the base dtype. Exponents must be finite integral
values with signed magnitude at most `u64::MAX`; fractional exponents are
`Unrepresentable`, and unsupported magnitudes are `Unsupported`. Negative powers
succeed only when the integer result is exact (bases ±1); zero to a negative power
errors. Dynamic `Number` retains integer evaluation for nonnegative integral exponents,
even when represented as floats, and promotes fractional or negative powers to
floats. Binary math chooses computation precision before converting its inputs;
a wider logarithm base no longer follows an intermediate f32 conversion.
Integer exponent inputs retain their full width, including for floating/complex
bases, so odd/even parity is not lost to floating conversion.

Floating `round` and `try_round` return floating values, preserving signed zero.
Use a checked cast when an integer is required. Boolean numerical functions use
0/1, while Boolean logical arithmetic remains unchanged.

Nonempty numeric aggregates start with their first element and consume inputs
left-to-right without zero shortcuts. Checked aggregates stop at the first error;
empty aggregates retain their existing identities. Concrete Boolean aggregates
retain logical short-circuiting.

### Explicit wrapping

`Int`, `UInt`, and `Number` provide inherent `wrapping_add`, `wrapping_sub`,
`wrapping_mul`, `wrapping_div`, `wrapping_rem`, `wrapping_abs`, `wrapping_pow`,
`wrapping_sum`, and `wrapping_product` methods. They return values directly,
without changing the ordinary or checked API or adding requirements to custom
numeric traits.

Wrapping changes integer overflow behavior, not promotion. Mixed widths and
signed/unsigned operands select the same result type as ordinary arithmetic.
Integer calculations wrap into that destination after evaluating the operands;
floating/complex results use ordinary IEEE/backend behavior. Boolean arithmetic
retains its logical rules. Explicit wrapping is independent of the application's
Rust overflow settings.

Integer division and remainder by zero panic, as with Rust's wrapping methods.
Signed minimum divided by minus one wraps to the minimum, and its remainder is
zero. Use `try_div`/`try_rem` for structured failures instead. Floating division by
zero retains IEEE behavior.

Typed integer `wrapping_pow` takes a `u64` exponent and retains the base dtype.
Dynamic `Number::wrapping_pow` takes `Number`: nonnegative integer powers wrap,
while negative or fractional powers follow ordinary promotion to floats. Complex
powers use the existing backend. `wrapping_sum` and `wrapping_product` seed from
the first value, consume left-to-right without zero shortcuts, and keep the
ordinary empty identities. No destination dtype argument is required.

For example, `Number::from(127i8).wrapping_add(Number::from(1u8))` is `128i16`,
while two I8 operands wrap to `-128i8`. Floating functions remain `sin`, `exp`,
`round`, and their checked counterparts; there is no parallel set of numerical
methods or mask-valued comparisons. Scalar/key equality and hashing remain
canonical, including NaNs.

ha-ndarray retains its own elementwise contract. Its integer zero divisors return
zero; scalar wrapping methods panic. Typed array powers retain their dtype for
negative exponents; dynamic scalar powers can promote. Array comparisons use
IEEE predicates, whereas scalar/key equality canonicalizes NaNs. Conformance
tests check shared behavior and assert these differences explicitly.

### Migration

- Checked real math uses pinned software kernels; final rounding can differ from
  native math. Portability vectors run in Linux and Windows CI; a local Linux pass
  does not by itself establish Windows agreement.
- Checked complex transcendentals now support the certified subset above and
  reject other inputs before evaluation.
- Integral floating exponents preserve integer bases and dtypes across ordinary,
  checked and wrapping powers. Logarithms prepare both arguments before conversion.

- The fixed-dtype scalar interface has been removed. Use ordinary numerical
  methods and casts, checked methods, or explicit wrapping arithmetic. Mixed
  numeric representations are supported through promotion.

- Replace implicit `UInt`→`Int`/`usize` conversions with `TryFrom` when loss is an
  error, or explicit `CastFrom` when the old lossy behavior is intended.
- Floating rounding now returns Float; add a checked integer cast if needed.
- Wildcard class casts now use source widths consistently with `into_type`;
  request a concrete target to fix a width.
- Boolean→Int uses I16, and Boolean trigonometry evaluates 0/1.
- Nonempty numeric aggregates preserve their first element's dtype until a later
  operand promotes it. Empty identities are unchanged.
- Custom numeric implementations must provide ordinary as well as checked
  methods. Generic logarithms no longer require `Float: From<N>`.

Equality, hashing, collation, codecs, and persisted encodings are unchanged by
these numerical API changes. ha-ndarray's production operations and kernels retain
their existing contract.

### WASM validation

Default and all-feature Rust libraries compile for `wasm32-unknown-unknown`.
The focused `--test wasm` integration target runs the shared checked portability
vectors and overflow classification through the wasm-bindgen test runner in Node.
The existing native portability test uses the same assertions. This verifies the
selected vectors rather than every operation on every WASM host. No production
JavaScript dependency or numerical policy is introduced by the tests.
